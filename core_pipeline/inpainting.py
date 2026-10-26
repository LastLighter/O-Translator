"""Inpainting backends: adaptive fast fill, ROI LaMa, and OpenCV fallback.

The desktop pipeline sees two very different kinds of background:

* solid/gradient UI panels, game chat boxes, and manga speech bubbles, where a
  local Telea fill or a median-colour fill is visually indistinguishable from a
  neural network at a fraction of the cost;
* textured artwork, where LaMa is worth its inference time.

``AutoInpainter`` therefore measures the colour variation of the pixels around
every text mask. Low-variance regions are filled locally (<15 ms). Only
high-variance regions are sent to LaMa, and only as small ``bbox + padding``
ROI patches, which are blended back into the main canvas. The same fast pass is
exposed to the pipeline so the UI can render a progressive preview while LaMa
is still running.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from core_pipeline.errors import InpaintingError, InpaintingUnavailableError
from core_pipeline.runtime import create_session
from core_pipeline.types import InpaintOutput, TextRegion

logger = logging.getLogger(__name__)


class Inpainter(Protocol):
    name: str

    def inpaint(self, image: Image.Image, mask: np.ndarray) -> InpaintOutput: ...


@dataclass(slots=True)
class InpaintRegion:
    """One text mask component and its surrounding-background statistics."""

    x: int
    y: int
    w: int
    h: int
    mask: np.ndarray
    background: tuple[int, int, int]
    stddev: float
    simple: bool

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))


@dataclass(slots=True)
class FastInpaint:
    """Result of the cheap first pass used by the progressive preview."""

    image: Image.Image
    plans: list[InpaintRegion] = field(default_factory=list)
    elapsed_ms: float = 0.0
    engine: str = "fast-telea"
    simple_regions: int = 0
    complex_regions: int = 0


def _as_binary_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Return an HxW uint8 mask of 0/255, resized if needed."""
    width, height = size
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _ellipse(size: int) -> np.ndarray:
    size = max(3, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _component_boxes(mask: np.ndarray, *, min_area: int = 4) -> list[tuple[int, int, int, int]]:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    boxes: list[tuple[int, int, int, int]] = []
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index][:5])
        if area < min_area or w <= 0 or h <= 0:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _box_union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _box_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(dx * dx + dy * dy) ** 0.5


def _merge_boxes(
    boxes: list[tuple[int, int, int, int]], *, gap: int = 2
) -> list[tuple[int, int, int, int]]:
    """Merge overlapping masks and glyph fragments separated by a few pixels."""
    expanded = [(x0 - gap, y0 - gap, x1 + gap, y1 + gap) for x0, y0, x1, y1 in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(expanded)):
            if merged:
                break
            for j in range(i + 1, len(expanded)):
                if _boxes_intersect(expanded[i], expanded[j]):
                    combined = _box_union(expanded[i], expanded[j])
                    expanded[i] = combined
                    expanded.pop(j)
                    merged = True
                    break
    return [
        (max(0, x0 + gap), max(0, y0 + gap), max(0, x1 - gap), max(0, y1 - gap))
        for x0, y0, x1, y1 in expanded
    ]


def _background_samples(
    crop: np.ndarray, local_mask: np.ndarray, *, ring_px: int
) -> np.ndarray | None:
    """Collect pixels in a ring just outside the mask (never inside it)."""
    ring_px = max(2, int(ring_px))
    dilated = cv2.dilate(local_mask, _ellipse(ring_px * 2 + 1))
    ring = (dilated > 0) & (local_mask == 0)
    samples = crop[ring]
    if samples.size >= 16 * 3:
        return samples.reshape(-1, 3)

    # Tiny masks near an image border may not have a usable ring. Fall back to
    # the crop border, which is still outside the text strokes in practice.
    if crop.size == 0:
        return None
    border = np.concatenate(
        [
            crop[0, :, :].reshape(-1, 3),
            crop[-1, :, :].reshape(-1, 3),
            crop[:, 0, :].reshape(-1, 3),
            crop[:, -1, :].reshape(-1, 3),
        ]
    )
    if border.size >= 8 * 3:
        return border.reshape(-1, 3)
    return None


def _plan_from_box(
    rgb: np.ndarray,
    binary: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    padding: int,
    ring_px: int,
    stddev_threshold: float,
) -> InpaintRegion | None:
    height, width = binary.shape[:2]
    x0 = max(0, box[0] - padding)
    y0 = max(0, box[1] - padding)
    x1 = min(width, box[2] + padding)
    y1 = min(height, box[3] + padding)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    local_mask = np.ascontiguousarray(binary[y0:y1, x0:x1])
    if not np.any(local_mask):
        return None
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    samples = _background_samples(crop, local_mask, ring_px=ring_px)
    if samples is None or len(samples) < 4:
        # Insufficient context: let LaMa handle it rather than guessing a fill.
        median = np.median(crop.reshape(-1, 3), axis=0)
        return InpaintRegion(
            x=x0,
            y=y0,
            w=x1 - x0,
            h=y1 - y0,
            mask=local_mask,
            background=tuple(int(round(float(v))) for v in median),  # type: ignore[arg-type]
            stddev=float("inf"),
            simple=False,
        )

    values = samples.astype(np.float32)
    stddev = float(np.mean(np.std(values, axis=0)))
    if stddev > stddev_threshold:
        # Only pay for the robust percentile pass when the direct measurement
        # says the region is textured; solid UI panels take the fast path.
        low = np.percentile(values, 5, axis=0)
        high = np.percentile(values, 95, axis=0)
        clipped = np.clip(values, low, high)
        stddev = float(np.mean(np.std(clipped, axis=0)))
    median = np.median(values, axis=0)
    return InpaintRegion(
        x=x0,
        y=y0,
        w=x1 - x0,
        h=y1 - y0,
        mask=local_mask,
        background=tuple(int(round(float(v))) for v in median),  # type: ignore[arg-type]
        stddev=stddev,
        simple=stddev <= stddev_threshold,
    )


def plan_inpaint_regions(
    image: Image.Image,
    mask: np.ndarray,
    *,
    padding: int = 16,
    ring_px: int = 6,
    stddev_threshold: float = 24.0,
    merge_gap: int = 3,
    regions: list[TextRegion] | None = None,
) -> tuple[np.ndarray, list[InpaintRegion]]:
    """Return ``(rgb, plans)`` for an image/mask pair.

    ``regions`` is accepted for forward compatibility (callers may eventually
    pass OCR boxes to avoid component segmentation), but the mask itself is the
    source of truth so glyph-aware masks work as well.
    """
    del regions  # currently informational only; the mask is authoritative
    rgb = np.asarray(image.convert("RGB"))
    binary = _as_binary_mask(mask, image.size)
    if not np.any(binary):
        return rgb, []
    boxes = _merge_boxes(_component_boxes(binary), gap=max(1, merge_gap))
    plans: list[InpaintRegion] = []
    for box in boxes:
        plan = _plan_from_box(
            rgb,
            binary,
            box,
            padding=max(0, int(padding)),
            ring_px=max(2, int(ring_px)),
            stddev_threshold=float(stddev_threshold),
        )
        if plan is not None:
            plans.append(plan)
    return rgb, plans


def _composite_patch(
    canvas: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray,
    *,
    roi: tuple[int, int],
) -> None:
    """Alpha-blend a patch back into ``canvas`` only where ``mask`` is set."""
    x0, y0 = roi
    h, w = mask.shape[:2]
    target = canvas[y0 : y0 + h, x0 : x0 + w]
    if target.shape[:2] != patch.shape[:2] or target.size == 0:
        return
    alpha = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
    alpha = np.clip((alpha - 0.10) / 0.90, 0.0, 1.0)[..., None]
    blended = patch.astype(np.float32) * alpha + target.astype(np.float32) * (1.0 - alpha)
    target[:] = np.clip(blended, 0, 255).round().astype(np.uint8)


def _composite_seamless(
    canvas: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray,
    *,
    roi: tuple[int, int],
) -> bool:
    """Try OpenCV's Poisson blend, returning False when the geometry is unsafe."""
    x0, y0 = roi
    h, w = mask.shape[:2]
    target = canvas[y0 : y0 + h, x0 : x0 + w]
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or patch.shape[:2] != (h, w):
        return False
    center = (int((int(xs.min()) + int(xs.max())) / 2), int((int(ys.min()) + int(ys.max())) / 2))
    if not (0 <= center[0] < w and 0 <= center[1] < h):
        return False
    if mask[center[1], center[0]] == 0:
        # seamlessClone needs the seed inside the mask; pick the closest
        # masked pixel to the bounding-box centre for L-shaped regions.
        nearest = int(np.argmin((xs - center[0]) ** 2 + (ys - center[1]) ** 2))
        center = (int(xs[nearest]), int(ys[nearest]))
    try:
        blended = cv2.seamlessClone(
            patch, target, mask, center, cv2.NORMAL_CLONE
        )
    except cv2.error:
        return False
    target[:] = blended
    return True


def _fast_fill(
    canvas: np.ndarray,
    rgb: np.ndarray,
    plan: InpaintRegion,
    *,
    solid_stddev: float,
    solid_only: bool = False,
) -> None:
    if plan.w <= 0 or plan.h <= 0:
        return
    patch = np.ascontiguousarray(rgb[plan.y : plan.bottom, plan.x : plan.right])
    mask = plan.mask
    if patch.size == 0 or not np.any(mask):
        return
    coverage = float(np.count_nonzero(mask)) / float(mask.size)
    if solid_only or plan.stddev <= solid_stddev or coverage > 0.92:
        # Hard fill is both faster and closer to the "solid box" preview the
        # UI shows; the mask is already dilated, so no feather is needed.
        target = canvas[plan.y : plan.bottom, plan.x : plan.right]
        target[mask > 0] = plan.background
        return
    try:
        patch = cv2.inpaint(patch, mask, 3, cv2.INPAINT_TELEA)
    except cv2.error:
        patch = patch.copy()
        patch[mask > 0] = plan.background
    _composite_patch(canvas, patch, mask, roi=(plan.x, plan.y))


def _paste_plan_mask(
    destination: np.ndarray,
    plan: InpaintRegion,
    *,
    roi: tuple[int, int, int, int],
) -> None:
    """Paste the intersection of a plan mask into a ROI-sized mask."""
    rx0, ry0, rx1, ry1 = roi
    ix0, iy0 = max(rx0, plan.x), max(ry0, plan.y)
    ix1, iy1 = min(rx1, plan.right), min(ry1, plan.bottom)
    if ix1 <= ix0 or iy1 <= iy0:
        return
    source = plan.mask[iy0 - plan.y : iy1 - plan.y, ix0 - plan.x : ix1 - plan.x]
    target = destination[iy0 - ry0 : iy1 - ry0, ix0 - rx0 : ix1 - rx0]
    np.maximum(target, source, out=target)


def _group_plans(
    plans: list[InpaintRegion],
    *,
    width: int,
    height: int,
    max_patches: int,
) -> list[tuple[int, int, int, int, list[InpaintRegion]]]:
    """Cluster complex regions into a bounded number of LaMa ROIs."""
    if not plans:
        return []
    groups: list[list[InpaintRegion]] = [[plan] for plan in plans]
    boxes = [(plan.x, plan.y, plan.right, plan.bottom) for plan in plans]

    # Union overlapping/nearby regions. Plan boxes already include padding, so
    # intersection is enough to keep a single speech bubble or UI row together.
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            if merged:
                break
            for j in range(i + 1, len(boxes)):
                if _boxes_intersect(boxes[i], boxes[j]):
                    boxes[i] = _box_union(boxes[i], boxes[j])
                    groups[i].extend(groups[j])
                    boxes.pop(j)
                    groups.pop(j)
                    merged = True
                    break

    # If a dense page produces dozens of tiny clusters, merge the closest pairs
    # instead of paying one full LaMa inference per cluster.
    limit = max(1, int(max_patches))
    while len(groups) > limit:
        best: tuple[float, int, int] | None = None
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                gap = _box_gap(boxes[i], boxes[j])
                if best is None or gap < best[0]:
                    best = (gap, i, j)
        if best is None:
            break
        _gap, i, j = best
        boxes[i] = _box_union(boxes[i], boxes[j])
        groups[i].extend(groups[j])
        boxes.pop(j)
        groups.pop(j)

    output: list[tuple[int, int, int, int, list[InpaintRegion]]] = []
    for box, group in zip(boxes, groups, strict=False):
        x0 = max(0, int(box[0]))
        y0 = max(0, int(box[1]))
        x1 = min(width, int(box[2]))
        y1 = min(height, int(box[3]))
        if x1 <= x0 or y1 <= y0:
            continue
        output.append((x0, y0, x1, y1, group))
    return output


def _lama_candidates(model_path: str | Path) -> list[Path]:
    path = Path(model_path)
    stem, suffix = path.stem, path.suffix
    for marker in ("_int8", "_fp16", "_sim", "_optimized"):
        if stem.endswith(marker):
            stem = stem[: -len(marker)]
            break
    names = [
        f"{stem}_sim{suffix}",
        f"{stem}_fp16{suffix}",
        f"{stem}_int8{suffix}",
        f"{stem}_optimized{suffix}",
        path.name,
    ]
    # An explicitly requested precision/variant always wins over auto siblings.
    if path.stem.endswith(("_int8", "_fp16", "_sim", "_optimized")):
        names = [path.name, *[name for name in names if name != path.name]]
    return [path.with_name(name) for name in dict.fromkeys(names)]


class OpenCVInpainter:
    """Fast, dependency-free fallback. Good for UI text, weaker on artwork."""

    name = "opencv-telea"

    def __init__(self, *, radius: int = 3) -> None:
        self.radius = max(1, radius)

    def inpaint(self, image: Image.Image, mask: np.ndarray) -> InpaintOutput:
        started = time.perf_counter()
        rgb = np.asarray(image.convert("RGB"))
        binary = _as_binary_mask(mask, image.size)
        try:
            # OpenCV operates directly on RGB data here; for inpainting only the
            # spatial mask matters and channel order stays consistent.
            result = cv2.inpaint(rgb, binary, self.radius, cv2.INPAINT_TELEA)
        except Exception as exc:
            raise InpaintingError(
                "OpenCV inpainting failed.", details={"error": str(exc)}
            ) from exc
        elapsed = (time.perf_counter() - started) * 1000.0
        return InpaintOutput(
            image=Image.fromarray(result, mode="RGB"),
            engine=self.name,
            elapsed_ms=elapsed,
            strategy="full-image-telea",
        )

    def fast_pass(
        self,
        image: Image.Image,
        mask: np.ndarray,
        regions: list[TextRegion] | None = None,
    ) -> FastInpaint:
        """Compatibility fast pass: OpenCV is already the cheap backend."""
        started = time.perf_counter()
        output = self.inpaint(image, mask)
        return FastInpaint(
            image=output.image,
            plans=[],
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            engine=self.name,
            simple_regions=0,
            complex_regions=0,
        )

    def inpaint_with_fast(
        self,
        image: Image.Image,
        mask: np.ndarray,
        regions: list[TextRegion] | None,
        fast: FastInpaint,
    ) -> InpaintOutput:
        return InpaintOutput(
            image=fast.image,
            engine=fast.engine,
            elapsed_ms=fast.elapsed_ms,
            strategy="full-image-telea",
        )


class LamaOnnxInpainter:
    """LaMa (WACV 2022) inference through ONNX Runtime.

    The loader tolerates the common ONNX export variants:
      * two inputs (``image`` + ``mask``) or one combined input
      * dynamic or fixed spatial dimensions
      * float image inputs normalized to [0,1], [-1,1], or ImageNet stats
    """

    name = "lama-onnx"

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        max_side: int = 1024,
        normalization: str = "zero_one",
        threads: int = 0,
        providers: str | None = None,
        openvino_device: str = "CPU",
        dml_device_id: int = 0,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.max_side = max(256, int(max_side))
        self.normalization = normalization
        self.threads = max(0, int(threads))
        self.providers_requested = providers
        self.openvino_device = openvino_device
        self.dml_device_id = int(dml_device_id)
        self._session = None
        self._loaded_path: Path | None = None
        self._input_names: list[str] = []
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._input_shapes: list[list[object]] = []
        self._registered_providers: list[str] = []

    @staticmethod
    def is_available(model_path: str | Path) -> bool:
        if not importlib.util.find_spec("onnxruntime"):
            return False
        return any(path.is_file() for path in _lama_candidates(model_path))

    @staticmethod
    def default_threads() -> int:
        """Threads for ORT CPU inference.

        The exported LaMa graph is dominated by FFT/Einsum subgraphs that stop
        scaling after a few threads; letting ONNX Runtime grab every logical
        core on a hybrid CPU makes inference 1.5-2x slower (and starves the OCR
        thread). Four threads is the measured sweet spot on 8-16 thread CPUs.
        """
        cpus = os.cpu_count() or 4
        return max(2, min(4, cpus // 2))

    def candidate_paths(self) -> list[Path]:
        return _lama_candidates(self.model_path)

    def optimized_path(self) -> Path:
        """Prefer an optimized sibling when one exists.

        ``onnxsim`` folding cuts load time from ~35 s to ~2 s. Quantized or
        FP16 siblings are preferred only when the caller explicitly selected a
        precision-named path (``_int8``/``_fp16``); otherwise the accurate
        folded FP32 graph wins on CPU.
        """
        candidates = self.candidate_paths()
        for candidate in candidates:
            try:
                if candidate.is_file() and candidate.stat().st_size >= 10 * 1024 * 1024:
                    return candidate
            except OSError:
                continue
        return self.model_path

    def _load(self) -> None:
        if self._session is not None:
            return
        # The startup warm-up thread and a job can reach this at the same time;
        # creating two ONNX sessions doubles both the memory and the cold-start
        # cost (the 198 MB model is graph-optimized on creation).
        with self._load_lock:
            if self._session is not None:
                return
            candidates = [self.optimized_path(), *self.candidate_paths()]
            seen: set[str] = set()
            session = None
            inputs = []
            last_error: Exception | None = None
            load_path = self.model_path
            threads = self.threads or self.default_threads()
            for candidate in candidates:
                key = str(candidate).lower()
                if key in seen:
                    continue
                seen.add(key)
                if not candidate.is_file():
                    continue
                try:
                    session, registered = create_session(
                        str(candidate),
                        device=self.device,
                        explicit_providers=self.providers_requested,
                        threads=threads,
                        openvino_device=self.openvino_device,
                        dml_device_id=self.dml_device_id,
                    )
                    inputs = session.get_inputs()
                    if len(inputs) < 2:
                        raise InpaintingError(
                            "This ONNX model does not expose separate image and "
                            "mask inputs."
                        )
                    load_path = candidate
                    self._registered_providers = registered
                    break
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "LaMa graph %s could not be loaded: %s", candidate, exc
                    )
                    session = None
            if session is None:
                if not any(path.is_file() for path in candidates):
                    raise InpaintingUnavailableError(
                        f"LaMa ONNX model not found at {self.model_path}. "
                        "See models/README.md or run scripts/download_models.py."
                    )
                raise InpaintingError(
                    "LaMa ONNX session could not be created.",
                    details={
                        "error": str(last_error),
                        "path": str(self.model_path),
                    },
                ) from last_error

            # Publish the finished session only after validation; ``_session``
            # last so another thread never observes a partially initialized
            # loader.
            self._input_names = [item.name for item in inputs]
            self._input_shapes = [list(item.shape) for item in inputs]
            self._loaded_path = load_path
            self._session = session

        logger.info(
            "Loaded LaMa ONNX: %s (providers=%s, threads=%s, inputs=%s)",
            self._loaded_path.name if self._loaded_path else load_path.name,
            self._registered_providers,
            threads,
            self._input_names,
        )

    def _target_size(self, width: int, height: int) -> tuple[int, int]:
        """Choose input dimensions accepted by the exported graph."""
        shape = self._input_shapes[0] if self._input_shapes else []
        model_h = shape[2] if len(shape) >= 4 else None
        model_w = shape[3] if len(shape) >= 4 else None
        if isinstance(model_h, int) and isinstance(model_w, int) and model_h > 0 and model_w > 0:
            return model_w, model_h

        scale = min(1.0, self.max_side / float(max(width, height)))
        target_w = max(64, int(round(width * scale)))
        target_h = max(64, int(round(height * scale)))
        # LaMa's FFC blocks are happiest with dimensions divisible by 8.
        target_w = max(64, target_w - (target_w % 8))
        target_h = max(64, target_h - (target_h % 8))
        return target_w, target_h

    def _normalize(self, rgb: np.ndarray) -> np.ndarray:
        image = rgb.astype(np.float32) / 255.0
        if self.normalization == "minus_one_one":
            image = image * 2.0 - 1.0
        elif self.normalization == "imagenet":
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image = (image - mean) / std
        return np.transpose(image, (2, 0, 1))[None, ...]

    def _infer_rgb(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Run LaMa and return an RGB uint8 array at the input resolution."""
        self._load()
        assert self._session is not None
        original = np.ascontiguousarray(image_rgb)
        height, width = original.shape[:2]
        binary = _as_binary_mask(mask, (width, height))
        target_w, target_h = self._target_size(width, height)

        resized = cv2.resize(original, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(
            binary, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )
        image_tensor = self._normalize(resized)
        mask_tensor = (resized_mask.astype(np.float32) / 255.0)[None, None, ...]
        if len(self._input_shapes) >= 2 and len(self._input_shapes[1]) == 4:
            channels = self._input_shapes[1][1]
            if isinstance(channels, int) and channels > 1:
                mask_tensor = np.repeat(mask_tensor, channels, axis=1)

        feeds: dict[str, np.ndarray] = {}
        for index, name in enumerate(self._input_names[:2]):
            lowered = name.lower()
            if "mask" in lowered or index == 1:
                feeds[name] = mask_tensor.astype(np.float32)
            else:
                feeds[name] = image_tensor.astype(np.float32)

        with self._lock:
            try:
                outputs = self._session.run(None, feeds)
            except Exception as exc:
                raise InpaintingError(
                    "LaMa inference failed.",
                    details={"error": str(exc), "size": [target_w, target_h]},
                ) from exc

        output = outputs[0]
        if output.ndim == 4:
            output = output[0]
        output = np.transpose(output, (1, 2, 0))
        if output.shape[2] == 1:
            output = np.repeat(output, 3, axis=2)
        output = output[:, :, :3]
        output = output.astype(np.float32)
        if output.min() < -0.5:
            output = (output + 1.0) / 2.0
        if output.max() > 1.5:
            output = output / 255.0
        output = np.clip(output, 0.0, 1.0)
        output_rgb = (output * 255.0).round().astype(np.uint8)
        if (target_w, target_h) != (width, height):
            output_rgb = cv2.resize(
                output_rgb, (width, height), interpolation=cv2.INTER_LANCZOS4
            )
        return output_rgb

    def inpaint(self, image: Image.Image, mask: np.ndarray) -> InpaintOutput:
        started = time.perf_counter()
        original = np.asarray(image.convert("RGB"))
        height, width = original.shape[:2]
        binary = _as_binary_mask(mask, (width, height))
        output_rgb = self._infer_rgb(original, binary)

        # Feather only the outer 1-2 px. The text core is fully replaced, which
        # avoids ghosting while keeping patch boundaries invisible.
        smooth_mask = cv2.GaussianBlur(binary, (3, 3), 0).astype(np.float32) / 255.0
        alpha = np.clip((smooth_mask - 0.15) / 0.85, 0.0, 1.0)[..., None]
        composited = (
            output_rgb.astype(np.float32) * alpha
            + original.astype(np.float32) * (1.0 - alpha)
        )
        result = np.clip(composited, 0, 255).round().astype(np.uint8)
        elapsed = (time.perf_counter() - started) * 1000.0
        return InpaintOutput(
            image=Image.fromarray(result, mode="RGB"),
            engine=self.name,
            elapsed_ms=elapsed,
            strategy="full-image-lama",
        )

    def inpaint_patch(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Run LaMa on one ROI and return the filled RGB patch."""
        return self._infer_rgb(image_rgb, mask)


class AutoInpainter:
    """Adaptive fast fill + ROI LaMa with OpenCV fallback.

    The first pass fills every region with a local Telea/median approximation.
    That result is immediately usable as the progressive preview background.
    LaMa then refines only the textured ROIs and is blended into the fast canvas.
    """

    name = "auto"

    def __init__(
        self,
        *,
        engine: str,
        lama_model_path: str | Path,
        device: str = "auto",
        max_side: int = 1024,
        normalization: str = "zero_one",
        threads: int = 0,
        providers: str | None = None,
        openvino_device: str = "CPU",
        dml_device_id: int = 0,
        fast_stddev: float = 24.0,
        solid_stddev: float = 3.0,
        roi_padding: int = 16,
        max_roi_patches: int = 1,
        blend: str = "alpha",
    ) -> None:
        self.engine = engine
        self.lama_path = Path(lama_model_path)
        self.device = device
        self.max_side = max_side
        self.normalization = normalization
        self.threads = max(0, int(threads))
        self.providers = providers
        self.openvino_device = openvino_device
        self.dml_device_id = int(dml_device_id)
        self.fast_stddev = max(0.0, float(fast_stddev))
        self.solid_stddev = max(0.0, min(float(solid_stddev), self.fast_stddev))
        self.roi_padding = max(0, int(roi_padding))
        self.max_roi_patches = max(1, int(max_roi_patches))
        self.blend = blend if blend in {"alpha", "seamless"} else "alpha"
        self._lama_lock = threading.Lock()
        self.lama = (
            LamaOnnxInpainter(
                self.lama_path,
                device=device,
                max_side=max_side,
                normalization=normalization,
                threads=self.threads,
                providers=providers,
                openvino_device=openvino_device,
                dml_device_id=dml_device_id,
            )
            if LamaOnnxInpainter.is_available(self.lama_path)
            else None
        )
        self.opencv = OpenCVInpainter()
        if engine == "lama" and self.lama is None:
            raise InpaintingUnavailableError(
                f"LaMa was requested but {self.lama_path} is missing or "
                "onnxruntime is not installed."
            )

    def _resolve_lama(self) -> LamaOnnxInpainter | None:
        if self.lama is None and self.engine in {"auto", "lama"}:
            with self._lama_lock:
                if self.lama is None and LamaOnnxInpainter.is_available(
                    self.lama_path
                ):
                    logger.info("LaMa weights became available; enabling LaMa")
                    self.lama = LamaOnnxInpainter(
                        self.lama_path,
                        device=self.device,
                        max_side=self.max_side,
                        normalization=self.normalization,
                        threads=self.threads,
                        providers=self.providers,
                        openvino_device=self.openvino_device,
                        dml_device_id=self.dml_device_id,
                    )
        return self.lama

    def warm_up(self) -> None:
        """Initialize ONNX Runtime / oneDNN before the first user screenshot.

        The first LaMa inference is much slower than later ones because the
        runtime compiles kernels. A tiny dummy ROI in a background thread
        removes that one-off cost from the first real job.
        """
        lama = self._resolve_lama()
        if lama is None:
            return
        try:
            dummy = np.full((96, 96, 3), 210, dtype=np.uint8)
            mask = np.zeros((96, 96), dtype=np.uint8)
            mask[24:72, 24:72] = 255
            lama.inpaint_patch(dummy, mask)
            logger.info("LaMa warm-up complete")
        except Exception as exc:  # pragma: no cover - hardware specific
            logger.warning("LaMa warm-up failed: %s", exc)

    def fast_pass(
        self,
        image: Image.Image,
        mask: np.ndarray,
        regions: list[TextRegion] | None = None,
    ) -> FastInpaint:
        """Fill every region with a local Telea/median approximation."""
        started = time.perf_counter()
        rgb, plans = plan_inpaint_regions(
            image,
            mask,
            padding=self.roi_padding,
            stddev_threshold=self.fast_stddev,
            regions=regions,
        )
        canvas = rgb.copy()
        for plan in plans:
            # Complex regions are about to be replaced by LaMa. A solid median
            # fill is enough for the <400 ms preview and much cheaper than Telea
            # on a 200x500 caption block.
            _fast_fill(
                canvas,
                rgb,
                plan,
                solid_stddev=self.solid_stddev,
                solid_only=not plan.simple,
            )
        simple = sum(1 for plan in plans if plan.simple)
        complex_count = len(plans) - simple
        return FastInpaint(
            image=Image.fromarray(canvas, mode="RGB"),
            plans=plans,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            engine="fast-telea",
            simple_regions=simple,
            complex_regions=complex_count,
        )

    def inpaint(
        self,
        image: Image.Image,
        mask: np.ndarray,
        regions: list[TextRegion] | None = None,
    ) -> InpaintOutput:
        fast = self.fast_pass(image, mask, regions)
        return self.inpaint_with_fast(image, mask, regions, fast)

    def _refine_fast_fallback(
        self,
        image: Image.Image,
        plans: list[InpaintRegion],
        canvas: np.ndarray,
    ) -> None:
        """Use local Telea for complex regions when LaMa cannot run."""
        rgb = np.asarray(image.convert("RGB"))
        for plan in plans:
            _fast_fill(
                canvas,
                rgb,
                plan,
                solid_stddev=self.solid_stddev,
                solid_only=False,
            )

    def _run_lama_rois(        self,
        image: Image.Image,
        mask: np.ndarray,
        plans: list[InpaintRegion],
        canvas: np.ndarray,
        lama: LamaOnnxInpainter,
    ) -> tuple[np.ndarray, int, int]:
        width, height = image.size
        rgb = np.asarray(image.convert("RGB"))
        binary = _as_binary_mask(mask, image.size)
        groups = _group_plans(
            plans, width=width, height=height, max_patches=self.max_roi_patches
        )
        patches = 0
        refined_regions = 0
        for x0, y0, x1, y1, group in groups:
            patch_rgb = np.ascontiguousarray(rgb[y0:y1, x0:x1])
            patch_mask = np.ascontiguousarray(binary[y0:y1, x0:x1])
            if patch_rgb.size == 0 or not np.any(patch_mask):
                continue
            patch_output = lama.inpaint_patch(patch_rgb, patch_mask)
            complex_mask = np.zeros(patch_mask.shape, dtype=np.uint8)
            for plan in group:
                _paste_plan_mask(complex_mask, plan, roi=(x0, y0, x1, y1))
            if not np.any(complex_mask):
                continue
            if self.blend == "seamless":
                if not _composite_seamless(
                    canvas, patch_output, complex_mask, roi=(x0, y0)
                ):
                    _composite_patch(canvas, patch_output, complex_mask, roi=(x0, y0))
            else:
                _composite_patch(canvas, patch_output, complex_mask, roi=(x0, y0))
            patches += 1
            refined_regions += len(group)
        return canvas, patches, refined_regions

    def inpaint_with_fast(
        self,
        image: Image.Image,
        mask: np.ndarray,
        regions: list[TextRegion] | None,
        fast: FastInpaint,
    ) -> InpaintOutput:
        started = time.perf_counter()
        canvas = np.array(fast.image.convert("RGB"), copy=True)
        complex_plans = [plan for plan in fast.plans if not plan.simple]
        warnings: list[str] = []
        engine = fast.engine
        patches = 0
        lama_regions = 0

        if complex_plans:
            lama = self._resolve_lama()
            if lama is not None and self.engine in {"auto", "lama"}:
                try:
                    canvas, patches, lama_regions = self._run_lama_rois(
                        image, mask, complex_plans, canvas, lama
                    )
                    if patches:
                        engine = "hybrid-lama" if fast.simple_regions else "lama-roi"
                    else:
                        warnings.append(
                            "LaMa ROI contained no composable mask; kept the fast fill."
                        )
                        engine = "opencv-telea"
                except Exception as exc:
                    if self.engine == "lama":
                        raise
                    logger.warning("LaMa ROI refinement failed, keeping fast fill: %s", exc)
                    self._refine_fast_fallback(image, complex_plans, canvas)
                    warnings.append(
                        "LaMa ROI refinement failed; used local Telea for "
                        "textured regions."
                    )
                    engine = "opencv-telea"
            elif self.engine == "lama":
                raise InpaintingUnavailableError(
                    "LaMa weights are not available; cannot refine textured regions."
                )
            else:
                self._refine_fast_fallback(image, complex_plans, canvas)
                warnings.append(
                    "LaMa weights are not available yet; used the fast OpenCV "
                    "Telea path for textured regions. The server downloads them "
                    "in the background when OTR_LAMA_AUTO_DOWNLOAD is enabled, or "
                    "run `python scripts/download_models.py --lama`."
                )
                engine = "opencv-telea"
        elif self.engine in {"auto", "lama"} and self._resolve_lama() is None:
            # Preserve the explicit "LaMa is missing" signal even when every
            # region was simple enough to skip LaMa entirely.
            warnings.append(
                "LaMa weights are not available; used the fast local fill for all "
                "regions. Run `python scripts/download_models.py --lama` to enable "
                "textured-background refinement."
            )
            engine = "opencv-telea"

        elapsed = fast.elapsed_ms + (time.perf_counter() - started) * 1000.0
        return InpaintOutput(
            image=Image.fromarray(canvas, mode="RGB"),
            engine=engine,
            elapsed_ms=elapsed,
            warnings=warnings,
            fast_regions=fast.simple_regions,
            lama_regions=lama_regions,
            roi_patches=patches,
            strategy=(
                "fast-only"
                if not complex_plans
                else ("hybrid-roi" if patches else "fast-fallback")
            ),
        )


def build_inpainter(
    *,
    engine: str = "auto",
    lama_model_path: str | Path,
    device: str = "auto",
    max_side: int = 1024,
    normalization: str = "zero_one",
    threads: int = 0,
    providers: str | None = None,
    openvino_device: str = "CPU",
    dml_device_id: int = 0,
    fast_stddev: float = 24.0,
    solid_stddev: float = 3.0,
    roi_padding: int = 16,
    max_roi_patches: int = 1,
    blend: str = "alpha",
) -> Inpainter:
    if engine == "opencv":
        return OpenCVInpainter()
    return AutoInpainter(
        engine=engine,
        lama_model_path=lama_model_path,
        device=device,
        max_side=max_side,
        normalization=normalization,
        threads=threads,
        providers=providers,
        openvino_device=openvino_device,
        dml_device_id=dml_device_id,
        fast_stddev=fast_stddev,
        solid_stddev=solid_stddev,
        roi_padding=roi_padding,
        max_roi_patches=max_roi_patches,
        blend=blend,
    )
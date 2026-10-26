"""Mask generation for text removal.

OCR polygons are rasterized into a binary mask. When the original image is
available the polygon is refined into a glyph mask, so inpainting only replaces
lettering strokes instead of the entire rectangular box. That preserves
gradients, speed lines, and artwork that happens to sit behind the text.

A small dilation (3-7 px by default) removes anti-aliased outlines and prevents
halos around the inpainted glyphs.

UI chrome is never rasterized: ``metadata`` rows (handles, timestamps, social
headers) and ``keep`` rows (language-neutral counters) stay pixel-identical, so
the icons packed next to a count — or a verified badge inside a header — are
not swallowed by a rectangular mask.
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np
from PIL import Image

from core_pipeline.types import TextRegion


def build_mask(
    size: tuple[int, int],
    regions: Iterable[TextRegion],
    *,
    dilate_px: int = 7,
    extra_expand_px: int = 0,
    image: Image.Image | None = None,
    glyph_aware: bool = False,
) -> np.ndarray:
    """Return an ``H x W`` uint8 mask with 255 on text pixels.

    Args:
        size: ``(width, height)`` of the target image.
        regions: OCR regions whose polygons should be filled.
        dilate_px: Elliptical dilation radius. Kernel is ``2*n+1``.
        extra_expand_px: Additional per-box expansion before rasterization.
        image: Original page. When supplied together with ``glyph_aware`` the
            filled polygon is narrowed to the lettering strokes it contains.
        glyph_aware: Refine masks against ``image`` instead of filling boxes.
    """
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    if width <= 0 or height <= 0:
        return mask
    # The function walks the regions twice (chrome protection, then text), so
    # accept any iterable without consuming a one-shot generator.
    regions = list(regions)

    rgb = None
    if glyph_aware and image is not None:
        rgb = np.asarray(image.convert("RGB"))

    # Chrome boxes are protected from every mask operation, including the
    # vertical growth applied to merged captions. Without this the 14-18 px
    # growth of a merged paragraph reached into the username row / avatar and
    # the fast fill smeared it into a grey ghost.
    protected = np.zeros((height, width), dtype=np.uint8)
    for region in regions:
        if region.role not in {"metadata", "keep"}:
            continue
        points = np.asarray(region.polygon, dtype=np.float32)
        if points.ndim != 2 or len(points) < 3:
            continue
        cv2.fillPoly(protected, [np.rint(points).astype(np.int32)], 255)

    def _grow_vertically(region: TextRegion, px: int) -> None:
        """Grow a horizontal text mask up/down only.

        A merged caption can miss its first/last line above or below the union
        box, so the mask needs vertical slack. Horizontal slack, however, eats
        the avatar and the verified badge sitting next to the text.
        """
        pad = max(0, int(px))
        if pad <= 0:
            return
        box = region.bbox
        y0 = max(0, box.y - pad)
        y1 = min(height, box.bottom + pad)
        x0 = max(0, box.x)
        x1 = min(width, box.right)
        if x1 <= x0 or y1 <= y0:
            return
        cv2.rectangle(
            mask,
            (x0, y0),
            (max(x0, x1 - 1), max(y0, y1 - 1)),
            255,
            thickness=-1,
        )

    for region in regions:
        # Usernames/timestamps (``metadata``) and language-neutral counters
        # (``keep``) are chrome: their original pixels — including avatars,
        # badges and the icons sitting next to a count — must survive. Erasing
        # and redrawing them only risks swallowing those graphics.
        if region.role in {"metadata", "keep"}:
            continue
        points = np.asarray(region.polygon, dtype=np.float32)
        if points.ndim != 2 or len(points) < 3:
            continue
        if rgb is not None:
            glyph_mask = _glyph_mask(rgb, points)
            if glyph_mask is not None:
                x0, y0, patch = glyph_mask
                y1 = y0 + patch.shape[0]
                x1 = x0 + patch.shape[1]
                mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], patch)
                _grow_vertically(region, region.mask_expand_px)
                if extra_expand_px:
                    box = region.bbox.expanded(extra_expand_px, width, height)
                    cv2.rectangle(
                        mask,
                        (box.x, box.y),
                        (max(box.x, box.right - 1), max(box.y, box.bottom - 1)),
                        255,
                        thickness=-1,
                    )
                continue
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
        if region.mask_expand_px:
            _grow_vertically(region, region.mask_expand_px)
        if extra_expand_px:
            box = region.bbox.expanded(extra_expand_px, width, height)
            cv2.rectangle(
                mask,
                (box.x, box.y),
                (max(box.x, box.right - 1), max(box.y, box.bottom - 1)),
                255,
                thickness=-1,
            )

    if dilate_px > 0:
        # ``dilate_px`` is the elliptical kernel diameter, constrained to the
        # 3x3 to 7x7 range recommended for anti-aliased lettering.
        kernel_size = min(7, max(3, int(dilate_px)))
        if kernel_size % 2 == 0:
            kernel_size = min(7, kernel_size + 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    if np.any(protected):
        mask[protected > 0] = 0
    return mask


def _glyph_mask(
    rgb: np.ndarray, points: np.ndarray
) -> tuple[int, int, np.ndarray] | None:
    """Extract lettering strokes inside one OCR polygon.

    Returns ``(x, y, mask)`` for the local patch, or ``None`` when the region is
    too ambiguous to refine and the caller should fall back to filling the box.
    """
    height, width = rgb.shape[:2]
    xs = np.clip(points[:, 0], 0, width - 1)
    ys = np.clip(points[:, 1], 0, height - 1)
    x0, y0 = int(np.floor(xs.min())) - 2, int(np.floor(ys.min())) - 2
    x1, y1 = int(np.ceil(xs.max())) + 2, int(np.ceil(ys.max())) + 2
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    crop = rgb[y0:y1, x0:x1].astype(np.int16)
    local_points = np.rint(points - np.array([x0, y0], dtype=np.float32)).astype(
        np.int32
    )
    polygon = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(polygon, [local_points], 255)
    polygon = cv2.dilate(
        polygon, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
    )
    if np.count_nonzero(polygon) < 24:
        return None

    # Background is estimated from a ring just outside the polygon; artwork
    # inside the box is deliberately not used because it biases the estimate.
    dilate_bg = cv2.dilate(
        polygon, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1
    )
    ring = (dilate_bg > 0) & (polygon == 0)
    if np.count_nonzero(ring) >= 12:
        background = np.median(crop[ring], axis=0)
    else:
        border = np.concatenate(
            [
                crop[0, :, :].reshape(-1, 3),
                crop[-1, :, :].reshape(-1, 3),
                crop[:, 0, :].reshape(-1, 3),
                crop[:, -1, :].reshape(-1, 3),
            ]
        )
        background = np.median(border, axis=0)
    distance_bg = np.linalg.norm(crop - background, axis=2)
    inside = polygon > 0
    values = distance_bg[inside]
    if values.size == 0:
        return None
    # Foreground is the strongest-contrast cluster; text is normally the most
    # distinct ink in its own box.
    strong_cutoff = float(np.percentile(values, 82))
    fg_pixels = crop[inside & (distance_bg >= strong_cutoff)]
    if len(fg_pixels) < 8:
        return None
    foreground = np.median(fg_pixels, axis=0)
    distance_fg = np.linalg.norm(crop - foreground, axis=2)

    # Two complementary cues: pixels near the dominant ink color, and pixels
    # far from the local background (catches outlines and drop shadows).
    stroke = ((distance_fg < 58) | (distance_bg > 96)) & inside
    # Thin strokes may be a single pixel wide; close gaps before dilating so
    # anti-aliased cores are replaced completely.
    stroke_u8 = np.where(stroke, 255, 0).astype(np.uint8)
    stroke_u8 = cv2.morphologyEx(
        stroke_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    stroke_u8 = cv2.dilate(
        stroke_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )
    stroke_u8 = cv2.bitwise_and(stroke_u8, polygon)
    coverage = np.count_nonzero(stroke_u8) / max(1, np.count_nonzero(inside))
    # A near-empty result means the heuristic missed the lettering; a near-full
    # result means the box is too dense/small to separate ink from background.
    if coverage < 0.015 or coverage > 0.88:
        return None
    return x0, y0, stroke_u8


def mask_to_pil(mask: np.ndarray) -> Image.Image:
    """Convert a binary mask to a grayscale PIL image for previews."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return Image.fromarray(mask.astype(np.uint8), mode="L")


def mask_coverage(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box of all positive mask pixels, or None when empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
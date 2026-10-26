"""Image decoding, normalization, and encoding helpers."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from core_pipeline.errors import InvalidImageError

# Reject decompression bombs before Pillow allocates an enormous bitmap. The
# explicit per-request pixel check below is the primary guard; this is a final
# safety net for other call sites.
Image.MAX_IMAGE_PIXELS = 80_000_000


def decode_image(
    data: bytes,
    *,
    max_side: int = 2400,
    max_pixels: int = 40_000_000,
) -> tuple[Image.Image, tuple[int, int]]:
    """Decode bytes into an upright RGB image and return its source size.

    Oversized screenshots are downscaled to ``max_side`` while preserving the
    aspect ratio. The result is always 8-bit RGB and safe to pass to OpenCV.
    """
    if not data:
        raise InvalidImageError("Empty image payload.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            if width <= 0 or height <= 0:
                raise InvalidImageError("Image has invalid dimensions.")
            if width * height > max_pixels:
                raise InvalidImageError(
                    f"Image is too large ({width}x{height}); limit is {max_pixels:,} pixels.",
                    details={"width": width, "height": height, "max_pixels": max_pixels},
                )
            probe.load()
            image = ImageOps.exif_transpose(probe)
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(
            "Unsupported or corrupted image file. Use PNG, JPEG, WEBP, or BMP."
        ) from exc

    source_size = image.size
    image = image.convert("RGB")
    if max(image.size) > max_side:
        ratio = max_side / float(max(image.size))
        new_size = (
            max(1, int(round(image.width * ratio))),
            max(1, int(round(image.height * ratio))),
        )
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image, source_size


def normalize_processed_size(
    image: Image.Image, *, max_side: int
) -> tuple[Image.Image, tuple[int, int]]:
    """Downscale an already-decoded image, preserving aspect ratio."""
    if max(image.size) <= max_side:
        return image, image.size
    ratio = max_side / float(max(image.size))
    size = (
        max(1, int(round(image.width * ratio))),
        max(1, int(round(image.height * ratio))),
    )
    return image.resize(size, Image.Resampling.LANCZOS), size


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """RGB PIL image -> contiguous RGB uint8 array."""
    return np.ascontiguousarray(np.asarray(image.convert("RGB")))


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """RGB/BGR-agnostic uint8 array -> PIL image. Expects RGB channel order."""
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    return Image.fromarray(np.ascontiguousarray(array).astype(np.uint8), mode="RGB")


def resize_within(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    """Resize an image to fit ``max_side``; returns image and applied scale."""
    if max(image.size) <= max_side:
        return image, 1.0
    scale = max_side / float(max(image.size))
    size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    return image.resize(size, Image.Resampling.LANCZOS), scale


def save_png_atomic(image: Image.Image, path: Path) -> None:
    """Write a PNG atomically so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".png", dir=str(path.parent)
    )
    os.close(fd)
    try:
        image.save(temporary, format="PNG", optimize=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def encode_jpeg(image: Image.Image, quality: int = 90) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    import base64

    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"

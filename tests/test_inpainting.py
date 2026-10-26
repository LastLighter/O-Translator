from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from core_pipeline.errors import InpaintingUnavailableError
from core_pipeline.inpainting import (
    AutoInpainter,
    LamaOnnxInpainter,
    OpenCVInpainter,
    build_inpainter,
)
from core_pipeline.mask import build_mask
from core_pipeline.types import TextRegion

FIXTURE = Path(__file__).parent / "fixtures" / "dummy_lama.onnx"


def _test_image() -> tuple[Image.Image, np.ndarray]:
    image = Image.new("RGB", (160, 100), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 30, 120, 65), fill=(30, 30, 30))
    region = TextRegion(
        id=0,
        text="text",
        polygon=[(40, 30), (120, 30), (120, 65), (40, 65)],
    )
    return image, build_mask((160, 100), [region], dilate_px=3)


def test_opencv_inpainting_changes_masked_region():
    image, mask = _test_image()
    output = OpenCVInpainter().inpaint(image, mask)
    assert output.engine == "opencv-telea"
    assert output.image.size == image.size
    assert np.asarray(output.image)[47, 80].tolist() != [30, 30, 30]


def test_auto_inpainter_warns_when_lama_missing(tmp_path):
    missing = tmp_path / "nope.onnx"
    engine = AutoInpainter(engine="auto", lama_model_path=missing)
    image, mask = _test_image()
    output = engine.inpaint(image, mask)
    assert output.engine == "opencv-telea"
    assert output.warnings


def test_explicit_lama_without_weights_raises(tmp_path):
    with pytest.raises(InpaintingUnavailableError):
        build_inpainter(engine="lama", lama_model_path=tmp_path / "nope.onnx")
    with pytest.raises(InpaintingUnavailableError):
        LamaOnnxInpainter(tmp_path / "nope.onnx").inpaint(
            Image.new("RGB", (32, 32)), np.zeros((32, 32), dtype=np.uint8)
        )


def test_lama_prefers_the_simplified_sibling(tmp_path):
    original = tmp_path / "lama_fp32.onnx"
    simplified = tmp_path / "lama_fp32_sim.onnx"
    original.write_bytes(b"placeholder")
    with simplified.open("wb") as handle:
        handle.truncate(11 * 1024 * 1024)

    assert LamaOnnxInpainter(original).optimized_path() == simplified
    # Passing the simplified path directly is also a no-op.
    assert LamaOnnxInpainter(simplified).optimized_path() == simplified


def test_lama_default_threads_are_capped():
    threads = LamaOnnxInpainter.default_threads()
    assert 2 <= threads <= 4


@pytest.mark.skipif(not FIXTURE.is_file(), reason="dummy ONNX fixture missing")
def test_lama_falls_back_when_the_simplified_graph_is_broken(tmp_path):
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        pytest.skip("onnxruntime not installed")
    (tmp_path / "lama_fp32.onnx").write_bytes(FIXTURE.read_bytes())
    with (tmp_path / "lama_fp32_sim.onnx").open("wb") as handle:
        handle.truncate(11 * 1024 * 1024)
    image, mask = _test_image()
    output = LamaOnnxInpainter(tmp_path / "lama_fp32.onnx", max_side=256).inpaint(
        image, mask
    )
    # The corrupt optimized sibling is skipped and the valid original is used.
    assert output.engine == "lama-onnx"
    assert output.image.size == image.size


@pytest.mark.skipif(not FIXTURE.is_file(), reason="dummy ONNX fixture missing")
def test_lama_onnx_loader_with_dummy_graph():
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        pytest.skip("onnxruntime not installed")
    image, mask = _test_image()
    inpainter = LamaOnnxInpainter(FIXTURE, max_side=256)
    output = inpainter.inpaint(image, mask)
    assert output.engine == "lama-onnx"
    assert output.image.size == image.size
    # The dummy graph is Identity; the feathered composite keeps the frame.
    array = np.asarray(output.image)
    assert array.shape == (100, 160, 3)
    assert array[0, 0].tolist() == [245, 245, 245]
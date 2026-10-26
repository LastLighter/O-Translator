from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from core_pipeline.inpainting import (
    AutoInpainter,
    LamaOnnxInpainter,
    plan_inpaint_regions,
)
from core_pipeline.mask import build_mask
from core_pipeline.types import TextRegion

FIXTURE = Path(__file__).parent / "fixtures" / "dummy_lama.onnx"


def _region(size: tuple[int, int], box: tuple[int, int, int, int]) -> TextRegion:
    x0, y0, x1, y1 = box
    return TextRegion(
        id=0,
        text="text",
        polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
    )


def _solid_image() -> tuple[Image.Image, np.ndarray]:
    image = Image.new("RGB", (220, 140), (242, 244, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 40, 160, 90), fill=(30, 32, 40))
    mask = build_mask(
        image.size,
        [_region(image.size, (60, 40, 160, 90))],
        dilate_px=3,
    )
    return image, mask


def _textured_image(seed: int = 4) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, (160, 220, 3), dtype=np.uint8)
    image = Image.fromarray(array, mode="RGB")
    mask = np.zeros((160, 220), dtype=np.uint8)
    mask[48:108, 62:168] = 255
    return image, mask


def test_solid_background_is_classified_as_simple():
    image, mask = _solid_image()
    _rgb, plans = plan_inpaint_regions(image, mask, stddev_threshold=12.0)
    assert plans
    assert all(plan.simple for plan in plans)
    assert max(plan.stddev for plan in plans) < 12.0


def test_textured_background_is_classified_as_complex():
    image, mask = _textured_image()
    _rgb, plans = plan_inpaint_regions(image, mask, stddev_threshold=8.0)
    assert plans
    assert any(not plan.simple for plan in plans)
    assert max(plan.stddev for plan in plans) > 8.0


def test_fast_pass_skips_lama_for_solid_ui_background(tmp_path):
    image, mask = _solid_image()
    inpainter = AutoInpainter(
        engine="auto",
        lama_model_path=tmp_path / "missing.onnx",
    )
    fast = inpainter.fast_pass(image, mask)
    assert fast.simple_regions >= 1
    assert fast.complex_regions == 0
    output = inpainter.inpaint_with_fast(image, mask, [], fast)
    assert output.engine == "opencv-telea"  # missing-LaMa legacy signal
    assert output.strategy == "fast-only"
    array = np.asarray(output.image)
    assert array[65, 110].tolist() != [30, 32, 40]


def test_roi_lama_path_uses_patch_and_blends_back():
    if not FIXTURE.is_file():
        return
    image, mask = _textured_image()
    inpainter = AutoInpainter(
        engine="auto",
        lama_model_path=FIXTURE,
        fast_stddev=0.0,
        roi_padding=12,
        max_roi_patches=2,
    )
    fast = inpainter.fast_pass(image, mask)
    assert fast.complex_regions >= 1
    output = inpainter.inpaint_with_fast(image, mask, [], fast)
    assert output.engine == "lama-roi"
    assert output.roi_patches >= 1
    assert output.lama_regions >= 1
    assert output.image.size == image.size


def test_roi_grouping_is_bounded():
    image, mask = _textured_image()
    inpainter = AutoInpainter(
        engine="auto",
        lama_model_path=FIXTURE,
        fast_stddev=0.0,
        max_roi_patches=1,
    )
    fast = inpainter.fast_pass(image, mask)
    # One mask component still produces one ROI; the important contract is that
    # a dense page can never exceed the configured patch budget.
    from core_pipeline.inpainting import _group_plans

    plans = [plan for plan in fast.plans if not plan.simple]
    groups = _group_plans(
        plans,
        width=image.width,
        height=image.height,
        max_patches=1,
    )
    assert len(groups) <= 1


def test_empty_mask_has_no_regions(tmp_path):
    image = Image.new("RGB", (80, 60), (255, 255, 255))
    mask = np.zeros((60, 80), dtype=np.uint8)
    inpainter = AutoInpainter(
        engine="auto",
        lama_model_path=tmp_path / "missing.onnx",
    )
    fast = inpainter.fast_pass(image, mask)
    assert fast.plans == []
    output = inpainter.inpaint_with_fast(image, mask, [], fast)
    assert output.image.size == image.size
    assert output.fast_regions == 0


def test_lama_candidate_order_prefers_explicit_precision(tmp_path):
    fp16 = tmp_path / "lama_fp32_fp16.onnx"
    fp16.write_bytes(b"x" * (11 * 1024 * 1024))
    inpainter = LamaOnnxInpainter(fp16)
    assert inpainter.optimized_path() == fp16

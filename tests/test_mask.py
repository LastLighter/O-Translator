from __future__ import annotations

import numpy as np

from core_pipeline.mask import build_mask, mask_bbox, mask_coverage, mask_to_pil
from core_pipeline.types import TextRegion


def _region(points):
    return TextRegion(id=0, text="hello", polygon=points)


def test_mask_fills_polygon_and_shape():
    region = _region([(10, 10), (60, 10), (60, 40), (10, 40)])
    mask = build_mask((100, 80), [region], dilate_px=0)
    assert mask.shape == (80, 100)
    assert mask.dtype == np.uint8
    assert mask[25, 35] == 255
    assert mask[5, 5] == 0
    assert mask_to_pil(mask).size == (100, 80)


def test_dilation_expands_mask():
    region = _region([(30, 30), (50, 30), (50, 50), (30, 50)])
    base = build_mask((100, 100), [region], dilate_px=0)
    dilated = build_mask((100, 100), [region], dilate_px=3)
    assert np.count_nonzero(dilated) > np.count_nonzero(base)
    assert mask_coverage(dilated) > mask_coverage(base)
    assert mask_bbox(base) == (30, 30, 51, 51)
    assert mask_bbox(dilated)[0] < 30


def test_empty_regions_produce_empty_mask():
    mask = build_mask((64, 64), [], dilate_px=5)
    assert not np.any(mask)
    assert mask_coverage(mask) == 0.0
    assert mask_bbox(mask) is None


def test_extra_expansion_covers_box():
    region = _region([(40, 40), (45, 40), (45, 45), (40, 45)])
    mask = build_mask((100, 100), [region], dilate_px=0, extra_expand_px=4)
    assert mask[36, 36] == 255
    assert mask[35, 35] == 0


def test_chrome_roles_are_not_masked():
    regions = [
        TextRegion(
            id=0,
            text="390",
            polygon=[(10, 10), (40, 10), (40, 30), (10, 30)],
            role="keep",
        ),
        TextRegion(
            id=1,
            text="ai_cosmic_horror_art 4小时",
            polygon=[(50, 10), (90, 10), (90, 30), (50, 30)],
            role="metadata",
        ),
        TextRegion(
            id=2,
            text="hello",
            polygon=[(10, 50), (40, 50), (40, 70), (10, 70)],
        ),
    ]
    mask = build_mask((100, 100), regions, dilate_px=0)
    assert mask[20, 25] == 0
    assert mask[20, 70] == 0
    assert mask[60, 25] == 255


def test_merged_caption_growth_is_vertical_only_and_spares_chrome():
    """The 14-18 px caption growth must not eat the username row/avatar."""
    regions = [
        TextRegion(
            id=0,
            text="レキ @reki13c · 36分钟",
            polygon=[(80, 10), (260, 10), (260, 40), (80, 40)],
            role="metadata",
        ),
        TextRegion(
            id=1,
            text="merged caption",
            polygon=[(70, 50), (300, 50), (300, 100), (70, 100)],
            mask_expand_px=18,
        ),
    ]
    mask = build_mask((400, 160), regions, dilate_px=0)
    # Chrome pixels are protected even though the caption's growth overlaps it.
    assert mask[25, 100] == 0
    assert mask[45, 100] == 255
    # No horizontal growth: the caption box starts at x=70 and its mask must too.
    assert mask[75, 69] == 0
    assert mask[75, 70] == 255


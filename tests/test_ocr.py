from __future__ import annotations

import numpy as np
from PIL import Image

from core_pipeline.ocr import (
    AutoOCRBackend,
    LanguageAwareOCRBackend,
    PaddleOCRBackend,
    contains_kana,
    deduplicate_regions,
    detect_text_direction,
    merge_stacked_lines,
    merge_title_pieces,
    normalize_paddle_lang,
    order_regions,
    repair_japanese_ocr,
)
from core_pipeline.types import OcrOutput, TextRegion


def test_normalize_paddle_lang():
    assert normalize_paddle_lang("ja") == "japan"
    assert normalize_paddle_lang("JP") == "japan"
    assert normalize_paddle_lang("zh-cn") == "ch"
    assert normalize_paddle_lang("auto", "ch") == "ch"
    assert normalize_paddle_lang("en") == "en"
    assert contains_kana("旅人よ")
    assert not contains_kana("那一天")


class _RoutingBackend:
    name = "paddle"

    def __init__(self, lang: str) -> None:
        self.lang = lang

    def read(self, image, **_options):
        return OcrOutput(
            regions=[
                TextRegion(
                    id=0,
                    text=f"{self.lang} text",
                    polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
                )
            ],
            engine=self.name,
            elapsed_ms=1.0,
        )


def test_language_aware_ocr_routes_and_caches():
    calls: list[str] = []

    def factory(lang: str) -> _RoutingBackend:
        calls.append(lang)
        return _RoutingBackend(lang)

    backend = LanguageAwareOCRBackend(factory, default_lang="ch", auto_fallback=False)
    assert backend.read(None, source_lang="ja", mode="auto").regions[0].text == "japan text"
    assert backend.read(None, source_lang="auto", mode="manga").regions[0].text == "japan text"
    assert calls == ["japan"]
    assert backend.read(None, source_lang="zh", mode="ui").regions[0].text == "ch text"
    assert calls == ["japan", "ch"]


def test_detect_text_direction():
    assert detect_text_direction("Hello there", 200, 30) == "horizontal"
    assert detect_text_direction("縦書きの文章です", 30, 240) == "vertical"
    assert detect_text_direction("あ", 20, 20) == "horizontal"


def test_language_aware_ocr_caches_identical_pages():
    class _CountingBackend(_RoutingBackend):
        def __init__(self, lang: str) -> None:
            super().__init__(lang)
            self.reads = 0

        def read(self, image, **_options):
            self.reads += 1
            return super().read(image)

    backend = _CountingBackend("ch")
    router = LanguageAwareOCRBackend(lambda lang: backend, auto_fallback=False)
    image = Image.new("RGB", (16, 16), (240, 240, 240))
    first = router.read(image, source_lang="zh", mode="ui")
    # Mutating the returned regions must not poison the cached copy.
    first.regions[0].text = "mutated"
    second = router.read(image, source_lang="zh", mode="ui")
    assert backend.reads == 1
    assert second.regions[0].text == "ch text"

    changed = Image.new("RGB", (16, 16), (240, 240, 241))
    router.read(changed, source_lang="zh", mode="ui")
    assert backend.reads == 2


def test_order_regions_ltr_rows():
    regions = [
        TextRegion(id=0, text="B", polygon=[(200, 20), (300, 20), (300, 50), (200, 50)]),
        TextRegion(id=1, text="A", polygon=[(20, 22), (120, 22), (120, 52), (20, 52)]),
        TextRegion(id=2, text="C", polygon=[(20, 90), (100, 90), (100, 120), (20, 120)]),
    ]
    ordered = order_regions(regions, mode="game", reading_order="ltr")
    assert [region.text for region in ordered] == ["A", "B", "C"]
    assert [region.id for region in ordered] == [0, 1, 2]


def test_order_regions_manga_rtl():
    regions = [
        TextRegion(
            id=0,
            text="left",
            polygon=[(20, 20), (80, 20), (80, 180), (20, 180)],
            direction="vertical",
        ),
        TextRegion(
            id=1,
            text="right",
            polygon=[(220, 20), (280, 20), (280, 180), (220, 180)],
            direction="vertical",
        ),
        TextRegion(
            id=2,
            text="right-low",
            polygon=[(220, 210), (280, 210), (280, 360), (220, 360)],
            direction="vertical",
        ),
    ]
    ordered = order_regions(regions, mode="manga", reading_order="auto")
    assert [region.text for region in ordered] == ["right", "right-low", "left"]


def test_deduplicate_same_text_same_box():
    a = TextRegion(
        id=0,
        text="same",
        confidence=0.8,
        polygon=[(10, 10), (100, 10), (100, 30), (10, 30)],
    )
    b = TextRegion(
        id=1,
        text="same",
        confidence=0.9,
        polygon=[(11, 10), (100, 10), (100, 30), (11, 30)],
    )
    result = deduplicate_regions([a, b])
    assert len(result) == 1
    assert result[0].confidence == 0.9


def test_parse_paddle_v3_result():
    backend = PaddleOCRBackend(min_confidence=0.4)
    raw = [
        {
            "rec_texts": ["Hello", "low confidence"],
            "rec_scores": [0.95, 0.1],
            "rec_polys": [
                np.array([[10, 10], [110, 10], [110, 40], [10, 40]]),
                np.array([[10, 60], [110, 60], [110, 90], [10, 90]]),
            ],
        }
    ]
    regions = backend._parse_results(raw, (200, 120))
    assert len(regions) == 1
    assert regions[0].text == "Hello"
    assert regions[0].bbox.w == 100


def test_parse_paddle_v2_result():
    backend = PaddleOCRBackend(min_confidence=0.1)
    raw = [
        [
            [
                [[10, 10], [120, 10], [120, 40], [10, 40]],
                ("Legacy", 0.88),
            ]
        ]
    ]
    regions = backend._parse_results(raw, (200, 100))
    assert len(regions) == 1
    assert regions[0].text == "Legacy"
    assert regions[0].confidence == 0.88


class _SharedModelBackend:
    name = "paddle"

    def __init__(self, lang: str) -> None:
        self.lang = lang
        self.reads = 0

    def model_signature(self):
        return ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec")

    def read(self, image, **_options):
        self.reads += 1
        return OcrOutput(regions=[], engine=self.name, elapsed_ms=1.0)


def test_language_aware_ocr_shares_backends_with_identical_models():
    created: list[_SharedModelBackend] = []

    def factory(lang: str) -> _SharedModelBackend:
        backend = _SharedModelBackend(lang)
        created.append(backend)
        return backend

    router = LanguageAwareOCRBackend(factory, auto_fallback=False)
    japan = router._backend("japan")
    chinese = router._backend("ch")
    assert japan is chinese
    # Both language keys exist, but only one initialized backend instance.
    assert len(router._backends) == 2
    assert len({id(backend) for backend in router._backends.values()}) == 1


class _RefiningBackend:
    name = "paddle"

    def __init__(self) -> None:
        self.reads = 0
        self.refines = 0

    def model_signature(self):
        return ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec")

    def can_refine_as_japanese(self) -> bool:
        return True

    def read(self, image, **_options):
        self.reads += 1
        return OcrOutput(
            regions=[
                TextRegion(
                    id=0,
                    text="旅人よ",
                    polygon=[(10, 10), (110, 10), (110, 40), (10, 40)],
                )
            ],
            engine=self.name,
            elapsed_ms=1.0,
        )

    def refine_regions(self, image, regions):
        self.refines += 1
        return [
            *regions,
            TextRegion(
                id=1,
                text="星の扉",
                polygon=[(10, 60), (110, 60), (110, 90), (10, 90)],
            ),
        ]


def test_auto_japanese_refine_reuses_the_recognized_page():
    backend = _RefiningBackend()
    router = LanguageAwareOCRBackend(lambda lang: backend, auto_fallback=True)
    output = router.read(None, source_lang="auto", mode="auto")  # type: ignore[arg-type]
    # One page OCR, no second full Japanese pass.
    assert backend.reads == 1
    assert backend.refines == 1
    assert [region.text for region in output.regions] == ["旅人よ", "星の扉"]
    assert any("in place" in warning for warning in output.warnings)


def test_refine_heuristic_ignores_confident_ui_labels():
    def region(text: str, width: int, height: int, confidence: float) -> TextRegion:
        return TextRegion(
            id=0,
            text=text,
            confidence=confidence,
            polygon=[(10, 10), (10 + width, 10), (10 + width, 10 + height), (10, 10 + height)],
        )

    # Crisp UI labels are short/wide or tiny, but the zoomed retry never splits
    # them; skipping it saves one crop detection each.
    assert not PaddleOCRBackend._needs_cluster_refine(region("CHAPTER IV", 69, 12, 0.97))
    assert not PaddleOCRBackend._needs_cluster_refine(region("设置", 32, 17, 1.0))
    # Merged manga titles and clipped caption lines score lower and still refine.
    assert PaddleOCRBackend._needs_cluster_refine(region("偽聖女", 204, 69, 0.82))
    assert PaddleOCRBackend._needs_cluster_refine(region("と呼ばれた", 41, 11, 0.50))
    assert not PaddleOCRBackend._needs_cluster_refine(region("07", 20, 12, 0.99))


class _EmptyOCR:
    name = "empty"

    def read(self, image):
        return OcrOutput(regions=[], engine=self.name, elapsed_ms=1.0)


class _GoodOCR:
    name = "good"

    def read(self, image):
        return OcrOutput(
            regions=[
                TextRegion(
                    id=0,
                    text="fallback",
                    polygon=[(10, 10), (100, 10), (100, 40), (10, 40)],
                )
            ],
            engine=self.name,
            elapsed_ms=1.0,
        )


def test_auto_ocr_falls_back_when_primary_returns_empty():
    backend = AutoOCRBackend([_EmptyOCR(), _GoodOCR()])
    output = backend.read(None)  # type: ignore[arg-type]
    assert output.engine == "good"
    assert output.regions[0].text == "fallback"
    assert any("empty" in warning for warning in output.warnings)


def test_auto_ocr_returns_empty_when_all_are_empty():
    backend = AutoOCRBackend([_EmptyOCR(), _EmptyOCR()])
    output = backend.read(None)  # type: ignore[arg-type]
    assert output.regions == []
    assert any("no text detected" in warning for warning in output.warnings)


def test_auto_ocr_warms_the_first_available_backend():
    warmed: list[str] = []

    class _WarmableOCR(_GoodOCR):
        def warm_up(self) -> None:
            warmed.append(self.name)

    backend = AutoOCRBackend([_WarmableOCR(), _EmptyOCR()])
    backend.warm_up()
    assert warmed == ["good"]


def test_repair_japanese_long_vowel_noise():
    assert repair_japanese_ocr("魔物狩りた1のし巻") == "魔物狩りたーのしー"
    assert repair_japanese_ocr("ヒヤツハ！`") == "ヒヤツハ！"
    assert repair_japanese_ocr("普通の中文") == "普通の中文"


def test_order_mixed_rtl_columns_and_ltr_caption():
    regions = [
        TextRegion(
            id=0,
            text="left",
            polygon=[(20, 20), (45, 20), (45, 240), (20, 240)],
            direction="vertical",
        ),
        TextRegion(
            id=1,
            text="right",
            polygon=[(70, 20), (95, 20), (95, 240), (70, 240)],
            direction="vertical",
        ),
        TextRegion(
            id=2,
            text="caption",
            polygon=[(20, 400), (200, 400), (200, 430), (20, 430)],
            direction="horizontal",
        ),
    ]
    ordered = order_regions(regions, mode="manga", reading_order="auto")
    assert [region.text for region in ordered] == ["right", "left", "caption"]
    assert [region.id for region in ordered] == [0, 1, 2]


def test_merge_stacked_caption_expands_mask():
    regions = [
        TextRegion(
            id=0,
            text="と呼ばれた",
            polygon=[(170, 628), (225, 628), (225, 638), (170, 638)],
            direction="horizontal",
        ),
        TextRegion(
            id=1,
            text="悪役に転生",
            polygon=[(174, 638), (228, 638), (228, 648), (174, 648)],
            direction="horizontal",
        ),
    ]
    merged = merge_stacked_lines(regions)
    assert len(merged) == 1
    assert merged[0].text == "と呼ばれた悪役に転生"
    assert merged[0].mask_expand_px >= 8


def test_merge_title_pieces_keeps_large_anchor():
    regions = [
        TextRegion(
            id=0,
            text="偽聖女",
            polygon=[(14, 631), (162, 631), (162, 691), (14, 691)],
            direction="horizontal",
        ),
        TextRegion(
            id=1,
            text="でした",
            polygon=[(155, 661), (206, 661), (206, 685), (155, 685)],
            direction="horizontal",
        ),
        TextRegion(
            id=2,
            text="caption",
            polygon=[(174, 630), (228, 630), (228, 647), (174, 647)],
            direction="horizontal",
        ),
    ]
    merged = merge_title_pieces(regions)
    title = next(region for region in merged if "偽聖女" in region.text)
    assert title.text == "偽聖女でした"
    # The union box keeps the original baseline width; large horizontal titles
    # are bottom-aligned during rendering so they stay off the caption ribbon.
    assert title.bbox.w == 192
    assert title.render_bbox is None


def test_merge_line_fragments_repairs_cuda_stack_boxes():
    """GPU detection split two lines into three overlapping tall boxes.

    They used to become two or three separate translation blocks whose glyphs
    were painted over each other. They must collapse into one layout block.
    """
    regions = [
        TextRegion(
            id=0,
            text="ネーム模写同じキャラが同",
            polygon=[(77, 57), (213, 57), (213, 100), (77, 100)],
            direction="horizontal",
        ),
        TextRegion(
            id=1,
            text="していて気付いたこと、ページ内で同じような構図、",
            polygon=[(166, 56), (477, 56), (477, 101), (166, 101)],
            direction="horizontal",
        ),
        TextRegion(
            id=2,
            text="同じようなポーズをしていて",
            polygon=[(467, 80), (714, 80), (714, 101), (467, 101)],
            direction="horizontal",
        ),
    ]
    assert _looks_fragmented(regions)
    merged = merge_line_fragments(regions)
    assert len(merged) == 1
    assert merged[0].bbox.model_dump() == {"x": 77, "y": 56, "w": 637, "h": 45}
    assert merged[0].text.startswith("ネーム模写")
    assert merged[0].text.endswith("同じようなポーズをしていて")


def test_merge_line_fragments_keeps_separate_toolbar_labels():
    regions = [
        TextRegion(
            id=0,
            text="AETHERIA",
            polygon=[(54, 40), (134, 40), (134, 59), (54, 59)],
            direction="horizontal",
        ),
        TextRegion(
            id=1,
            text="CHAPTER IV",
            polygon=[(165, 38), (239, 38), (239, 58), (165, 58)],
            direction="horizontal",
        ),
    ]
    assert not _looks_fragmented(regions)
    merged = merge_line_fragments(regions)
    assert [region.text for region in merged] == ["AETHERIA", "CHAPTER IV"]


def test_merge_stacked_lines_rejects_side_by_side_columns():
    """Two fragments from different rows overlap horizontally but do not stack."""
    regions = [
        TextRegion(
            id=0,
            text="していて気付いたこと、",
            polygon=[(166, 56), (357, 56), (357, 77), (166, 77)],
            direction="horizontal",
        ),
        TextRegion(
            id=1,
            text="ページ内で同じような構図、",
            polygon=[(250, 79), (471, 79), (471, 100), (250, 100)],
            direction="horizontal",
        ),
    ]
    merged = merge_stacked_lines(regions)
    assert len(merged) == 2


def test_title_merge_does_not_duplicate_prefix_glyph():
    wide = TextRegion(
        id=0,
        text="同じようなポーズをしていて",
        polygon=[(487, 80), (714, 80), (714, 101), (487, 101)],
        direction="horizontal",
    )
    tiny = TextRegion(
        id=1,
        text="同",
        polygon=[(471, 84), (493, 84), (493, 97), (471, 97)],
        direction="horizontal",
    )
    assert len(merge_title_pieces([tiny, wide])) == 2
    deduped = deduplicate_regions([tiny, wide])
    assert [region.text for region in deduped] == ["同じようなポーズをしていて"]


def test_resolve_overlapping_regions_collapses_overprinting_blocks():
    regions = [
        TextRegion(
            id=0,
            text="ネーム模写同じキャラが同",
            polygon=[(77, 57), (213, 57), (213, 100), (77, 100)],
            direction="horizontal",
        ),
        TextRegion(
            id=1,
            text="していて気付いたこと、ページ内で同じような構図、",
            polygon=[(166, 56), (477, 56), (477, 101), (166, 101)],
            direction="horizontal",
        ),
    ]
    resolved = resolve_overlapping_regions(regions)
    assert len(resolved) == 1
    assert resolved[0].bbox.model_dump() == {"x": 77, "y": 56, "w": 400, "h": 45}


def test_merge_paragraph_lines_reconstructs_wrapped_post():
    regions = [
        TextRegion(
            id=0,
            text="museforhumans>",
            confidence=0.9614,
            polygon=[(64, 3), (250, 3), (250, 22), (64, 22)],
            direction="horizontal",
        ),
        TextRegion(
            id=1,
            text="Muse",
            confidence=0.9994,
            polygon=[(245, 2), (325, 2), (325, 21), (245, 21)],
            direction="horizontal",
        ),
        TextRegion(
            id=2,
            text="1小时",
            confidence=0.9978,
            polygon=[(325, 1), (375, 1), (375, 21), (325, 21)],
            direction="horizontal",
        ),
        TextRegion(
            id=3,
            text="He paused the old movie mid-scene — 'I know that face.' One photo of the",
            confidence=0.9884,
            polygon=[(62, 32), (745, 32), (745, 54), (62, 54)],
            direction="horizontal",
        ),
        TextRegion(
            id=4,
            text="TV and Muse named the actor, the film, and the other role he was",
            confidence=0.9973,
            polygon=[(61, 57), (659, 57), (659, 80), (61, 80)],
            direction="horizontal",
        ),
        TextRegion(
            id=5,
            text="remembering him from. Ask your own Muse to be your couch co-pilot. I",
            confidence=0.9985,
            polygon=[(60, 84), (722, 84), (722, 107), (60, 107)],
            direction="horizontal",
        ),
        TextRegion(
            id=6,
            text="am Trill. Muse翻译",
            confidence=0.9645,
            polygon=[(62, 107), (242, 107), (242, 135), (62, 135)],
            direction="horizontal",
        ),
    ]
    merged = merge_paragraph_lines(regions, page_width=750)
    # The header labels stay separate; only the four body lines form one block.
    assert [region.text for region in merged[:3]] == [
        "museforhumans>",
        "Muse",
        "1小时",
    ]
    assert len(merged) == 4
    body = merged[3]
    assert body.text == (
        "He paused the old movie mid-scene — 'I know that face.' One photo of the "
        "TV and Muse named the actor, the film, and the other role he was "
        "remembering him from. Ask your own Muse to be your couch co-pilot. I "
        "am Trill. Muse翻译"
    )
    assert body.bbox.model_dump() == {"x": 60, "y": 32, "w": 685, "h": 103}
    assert body.source_line_height == 23.0
    # The union box is several lines tall; the budget must still assume normal
    # line height instead of one giant line.
    assert body.capacity_hint("zh") > 120


def test_merge_paragraph_lines_joins_cjk_without_spaces():
    regions = [
        TextRegion(
            id=0,
            text="これはとても長いテスト",
            polygon=[(10, 20), (210, 20), (210, 46), (10, 46)],
            direction="horizontal",
        ),
        TextRegion(
            id=1,
            text="です。",
            polygon=[(10, 50), (80, 50), (80, 76), (10, 76)],
            direction="horizontal",
        ),
    ]
    merged = merge_paragraph_lines(regions, page_width=400)
    assert len(merged) == 1
    assert merged[0].text == "これはとても長いテストです。"


def test_merge_paragraph_lines_keeps_stacked_ui_labels():
    regions = [
        TextRegion(
            id=0,
            text="File",
            polygon=[(10, 10), (70, 10), (70, 32), (10, 32)],
            direction="horizontal",
        ),
        TextRegion(
            id=1,
            text="Edit",
            polygon=[(10, 36), (74, 36), (74, 58), (10, 58)],
            direction="horizontal",
        ),
    ]
    merged = merge_paragraph_lines(regions, page_width=1200)
    assert len(merged) == 2


def test_merge_paragraph_lines_does_not_swallow_new_list_item():
    regions = [
        TextRegion(
            id=0,
            text="- first item with enough words",
            polygon=[(10, 10), (280, 10), (280, 34), (10, 34)],
            direction="horizontal",
        ),
        TextRegion(
            id=1,
            text="- second item with enough words",
            polygon=[(10, 38), (290, 38), (290, 62), (10, 62)],
            direction="horizontal",
        ),
    ]
    merged = merge_paragraph_lines(regions, page_width=400)
    assert len(merged) == 2

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from core_pipeline.types import BBox, TextRegion
from core_pipeline.typesetting import (
    FontBook,
    Renderer,
    _line_height,
    _wrap_cjk,
    _wrap_generic,
    analyze_text_style,
    contrast_ratio,
    extract_colors,
    fit_text,
    relative_luminance,
    wrap_text,
)


def test_fontbook_finds_a_font():
    book = FontBook()
    assert book.get(20) is not None
    assert book.get(20) is book.get(20)


def test_wrap_long_english_respects_width():
    book = FontBook()
    font = book.get(16)
    max_width = 140
    lines = wrap_text(
        "This is a fairly long sentence that must be wrapped cleanly across lines.",
        font,
        max_width,
    )
    assert len(lines) > 1
    assert all(font.getlength(line) <= max_width + 1 for line in lines)


def test_wrap_long_word_splits_when_narrower_than_token():
    book = FontBook()
    font = book.get(16)
    lines = wrap_text("Supercalifragilisticexpialidocious", font, 80)
    assert len(lines) > 1
    assert all(font.getlength(line) <= 80 + 1 for line in lines)


def test_fit_uses_available_box():
    book = FontBook()
    box = BBox(x=20, y=20, w=170, h=70)
    result = fit_text("Short translated line", box, book, target_lang="en")
    assert not result.overflow
    assert result.lines
    assert result.font_size >= 7
    assert result.height <= box.h


def test_vertical_layout_only_for_cjk_translation():
    book = FontBook()
    box = BBox(x=0, y=0, w=50, h=200)
    cjk = fit_text(
        "星の扉が開く",
        box,
        book,
        target_lang="zh",
        source_direction="vertical",
    )
    latin = fit_text(
        "The star gate opens",
        box,
        book,
        target_lang="zh",
        source_direction="vertical",
    )
    assert cjk.layout == "vertical"
    assert latin.layout == "horizontal"


def test_extract_colors_for_dark_panel():
    image = Image.new("RGB", (220, 90), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 15, 200, 75), fill=(25, 30, 38))
    draw.text((40, 32), "HELLO", fill=(245, 242, 234))
    foreground, background = extract_colors(image, BBox(x=20, y=15, w=180, h=60))
    assert contrast_ratio(foreground, background) >= 3.2
    assert sum(background) < 250


def test_cached_page_pixels_match_on_the_fly_conversion():
    image = Image.new("RGB", (240, 140), (250, 244, 226))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 210, 110), fill=(22, 34, 48))
    draw.text((50, 55), "CACHED", fill=(248, 245, 238))
    box = BBox(x=30, y=30, w=180, h=80)
    pixels = np.asarray(image.convert("RGB"))
    assert analyze_text_style(image, box, mode="manga", font_size=24) == (
        analyze_text_style(image, box, mode="manga", font_size=24, pixels=pixels)
    )
    assert extract_colors(image, box) == extract_colors(image, box, pixels=pixels)


def test_fast_cjk_wrap_matches_generic_wrapper():
    book = FontBook()
    samples = [
        "魔物狩りたーのしー",
        "何というかね……フフ",
        "这是翻译后的中文文本内容。",
        "偽聖女でした",
        "星の扉が開く、夜が明ける。",
    ]
    for size in (9, 14, 24, 40):
        font = book.get(size)
        for text in samples:
            for width in (40, 77, 140, 300):
                fast, verified = _wrap_cjk(text, font, width)
                assert verified
                assert fast == _wrap_generic(
                    text, font, width, allow_empty=True
                )


def test_fit_returns_the_largest_fitting_size():
    """Binary-search-style optimizations must not shrink the chosen font."""
    book = FontBook()
    box = BBox(x=0, y=0, w=81, h=206)
    text = "魔物狩りたーのしー"
    result = fit_text(
        text,
        box,
        book,
        target_lang="ja",
        min_size=9,
        font_scale=0.6,
        max_size=185,
    )
    assert not result.overflow
    max_w, max_h = box.w - 4, box.h - 4
    start = max(9, min(int(min(max_w, max_h) * 0.98 * 0.6), 185))
    # No larger size may fit; this is what the exhaustive downward scan found.
    for size in range(result.font_size + 1, start + 1):
        font = book.get(size)
        lines = wrap_text(text, font, max_w)
        measured = max(font.getlength(line) for line in lines)
        total_h = len(lines) * _line_height(font, size, tight=True)
        assert measured > max_w + 0.5 or total_h > max_h + 0.5


def test_renderer_produces_blocks_and_text(sample_image, sample_regions):
    renderer = Renderer(FontBook(), min_font_size=7)
    clean = sample_image.copy()
    # Simulate an inpainted plate by painting the two boxes.
    draw = ImageDraw.Draw(clean)
    draw.ellipse((540, 62, 716, 238), fill=(103, 86, 154))
    draw.rounded_rectangle((46, 58, 496, 156), radius=16, fill=(24, 29, 38))

    output = renderer.render(
        sample_image,
        clean,
        sample_regions,
        {0: "Hello, welcome to O-Translator!", 1: "Test text"},
        target_lang="en",
    )
    assert len(output.blocks) == 2
    assert output.blocks[0].translated_text.startswith("Hello")
    assert output.blocks[0].font_size >= 7
    assert output.blocks[0].lines
    assert output.image.size == sample_image.size
    assert output.image.tobytes() != clean.tobytes()


def test_renderer_handles_empty_text(sample_image):
    renderer = Renderer(FontBook())
    region = TextRegion(
        id=0,
        text="",
        polygon=[(10, 10), (100, 10), (100, 50), (10, 50)],
    )
    output = renderer.render(sample_image, sample_image, [region], {0: ""})
    assert output.blocks[0].translated_text == ""


def test_dark_artwork_uses_white_lettering_with_outline():
    image = Image.new("RGB", (120, 80), (25, 20, 30))
    draw = ImageDraw.Draw(image)
    draw.text((20, 30), "WHITE", fill=(250, 250, 250))
    fill, stroke, width, _ = analyze_text_style(
        image, BBox(x=10, y=10, w=100, h=60), mode="manga", font_size=24
    )
    assert fill == (255, 255, 255)
    assert stroke is not None
    assert width >= 1


def test_bright_caption_keeps_ink_and_adds_light_outline():
    image = Image.new("RGB", (200, 80), (250, 242, 220))
    draw = ImageDraw.Draw(image)
    draw.text((20, 30), "伪圣女", fill=(20, 50, 80))
    fill, stroke, width, _ = analyze_text_style(
        image, BBox(x=10, y=10, w=180, h=60), mode="manga", font_size=32
    )
    assert relative_luminance(fill) < 0.6
    assert stroke == (255, 255, 255)
    assert width == 1


def test_paragraph_uses_source_line_height_not_union_box():
    """A tall merged paragraph must not scale its Chinese text like a title."""
    image = Image.new("RGB", (600, 320), (18, 18, 24))
    region = TextRegion(
        id=0,
        text=(
            "なんかよくわからん伸びしたんで宣伝ぶら下げときます！！！"
            "イラストの演出動画ソフトを「無料」公開してます。"
            "さっきしました！い。イラストをパズルにした動画とかが作れるよ！ぜひ！"
        ),
        polygon=[(8, 95), (554, 95), (554, 294), (8, 294)],
        direction="horizontal",
        source_line_height=20.5,
    )
    output = Renderer(FontBook(), min_font_size=7).render(
        image,
        image,
        [region],
        {0: "不知怎么就火了，顺便宣传一下！！！插画演出视频软件「免费」公开中。"},
        target_lang="zh",
        mode="auto",
    )
    block = output.blocks[0]
    # The 199 px union box previously produced a 27 px font; the source lines
    # are ~20.5 px tall, so the translation must stay in that range.
    assert block.font_size <= 19
    assert block.line_height * len(block.lines) <= 80
    assert not block.overflow


def test_merged_tag_block_uses_source_scale_and_regular_weight():
    image = Image.new("RGB", (700, 420), (250, 248, 242))
    region = TextRegion(
        id=0,
        text="#cosmichorror #cthulhu #lovecraft #cosmic_horror_art",
        polygon=[(50, 50), (650, 50), (650, 270), (50, 270)],
        direction="horizontal",
        role="hashtag",
        source_line_height=24.0,
    )
    output = Renderer(FontBook(), min_font_size=7).render(
        image,
        image,
        [region],
        {0: "#宇宙恐怖 #克苏鲁 #洛夫克拉夫特 #星之彩 #伊格教团"},
        target_lang="zh",
        mode="auto",
    )
    block = output.blocks[0]
    # The 220 px-tall union box must not inflate a 24 px tag line.
    assert block.font_size <= 22
    assert block.font_weight == "regular"
    assert block.role == "hashtag"
    assert not block.overflow


def test_hashtag_capacity_hint_keeps_every_tag():
    text = (
        "#cosmichorror #cthulhu #darkfantasyart #surrealart #lovecraft "
        "#cosmichorror #cthulhu"
    )
    region = TextRegion(
        id=0,
        text=text,
        polygon=[(50, 50), (700, 50), (700, 76), (50, 76)],
        direction="horizontal",
        role="hashtag",
    )
    assert region.capacity_hint("zh") >= len(text) + 40


def test_renderer_preserves_source_edge_alignment():
    image = Image.new("RGB", (800, 180), (250, 248, 242))
    regions = [
        TextRegion(
            id=0,
            text="Left label",
            polygon=[(20, 30), (220, 30), (220, 66), (20, 66)],
            direction="horizontal",
        ),
        TextRegion(
            id=1,
            text="Centered label",
            polygon=[(300, 90), (500, 90), (500, 126), (300, 126)],
            direction="horizontal",
        ),
        TextRegion(
            id=2,
            text="Right label",
            polygon=[(600, 30), (770, 30), (770, 66), (600, 66)],
            direction="horizontal",
        ),
    ]
    output = Renderer(FontBook()).render(
        image,
        image,
        regions,
        {0: "左", 1: "中", 2: "右"},
        target_lang="zh",
        mode="auto",
    )
    assert [block.align for block in output.blocks] == ["left", "center", "right"]


def test_chrome_rows_are_not_redrawn():
    image = Image.new("RGB", (500, 120), (250, 248, 242))
    regions = [
        TextRegion(
            id=0,
            text="ai_cosmic_horror_art 4小时",
            polygon=[(40, 20), (360, 20), (360, 50), (40, 50)],
            direction="horizontal",
            role="metadata",
        ),
        TextRegion(
            id=1,
            text="390",
            polygon=[(40, 70), (90, 70), (90, 96), (40, 96)],
            direction="horizontal",
            role="keep",
        ),
    ]
    output = Renderer(FontBook(), min_font_size=7).render(
        image,
        image,
        regions,
        {0: "ai_cosmic_horror_art 4小时", 1: "390"},
        target_lang="zh",
    )
    assert output.blocks == []
    assert output.image.tobytes() == image.tobytes()
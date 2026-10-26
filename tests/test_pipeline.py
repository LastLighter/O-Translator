from __future__ import annotations

import asyncio
import time

import pytest

from core_pipeline.errors import InvalidImageError, TranslationError
from core_pipeline.inpainting import OpenCVInpainter
from core_pipeline.pipeline import TranslationPipeline, _display_metadata
from core_pipeline.translation import EchoTranslator
from core_pipeline.types import (
    InpaintOutput,
    OcrOutput,
    PipelineOptions,
    TextRegion,
    TranslationOutput,
)
from core_pipeline.typesetting import FontBook, Renderer


class FakeOCR:
    name = "fake-ocr"

    def __init__(self, regions):
        self.regions = regions

    def read(self, image):
        return OcrOutput(regions=self.regions, engine=self.name, elapsed_ms=1.0)


class FailingTranslator:
    name = "failing"
    model = "test"

    async def translate(self, regions, options):
        raise TranslationError("API exploded")

    async def aclose(self):
        return None


class RecordingTranslator:
    name = "recording"
    model = "test"

    def __init__(self):
        self.seen = None

    async def translate(self, regions, options):
        self.seen = regions
        return TranslationOutput(
            translations={region.id: f"EN {region.text}" for region in regions},
            provider=self.name,
            model=self.model,
            elapsed_ms=1.0,
        )

    async def aclose(self):
        return None


class SlowInpainter:
    name = "slow"
    delay = 0.3

    def inpaint(self, image, mask):
        time.sleep(self.delay)

        return InpaintOutput(image=image.copy(), engine=self.name, elapsed_ms=300)


class SlowTranslator:
    name = "slow"
    model = "test"
    delay = 0.3

    async def translate(self, regions, options):
        await asyncio.sleep(self.delay)
        return TranslationOutput(
            translations={region.id: region.text for region in regions},
            provider=self.name,
            model=self.model,
            elapsed_ms=300,
        )

    async def aclose(self):
        return None


def make_pipeline(ocr, *, translator=None, inpainter=None):
    return TranslationPipeline(
        ocr=ocr,
        inpainter=inpainter or OpenCVInpainter(),
        translator=translator or EchoTranslator(),
        renderer=Renderer(FontBook()),
        dilate_px=5,
    )


def test_pipeline_end_to_end(sample_png, sample_regions, options):
    translator = RecordingTranslator()
    pipeline = make_pipeline(FakeOCR(sample_regions), translator=translator)
    stages: list[str] = []
    result = asyncio.run(
        pipeline.run(
            sample_png,
            options,
            progress=lambda stage, progress, message: stages.append(stage),
        )
    )
    assert not result.empty
    assert len(result.regions) == 2
    assert len(result.blocks) == 2
    assert result.rendered.size == result.original.size
    assert result.clean.size == result.original.size
    assert result.translations[0].startswith("EN ")
    assert result.ocr_engine == "fake-ocr"
    assert "ocr" in stages and "rendering" in stages and stages[-1] == "complete"
    assert translator.seen is not None
    assert result.timings_ms["total_ms"] >= 0


def test_pipeline_does_not_render_overlapping_ocr_fragments(sample_png):
    """Fragmented GPU boxes must collapse into one block before typesetting."""
    regions = [
        TextRegion(
            id=0,
            text="レキ@reki13c·36分钟",
            confidence=0.98,
            polygon=[(72, 20), (300, 20), (300, 44), (72, 44)],
            direction="horizontal",
        ),
        TextRegion(
            id=1,
            text="ネーム模写同じキャラが同",
            confidence=0.99,
            polygon=[(77, 57), (213, 57), (213, 100), (77, 100)],
            direction="horizontal",
        ),
        TextRegion(
            id=2,
            text="していて気付いたこと、ページ内で同じような構図、",
            confidence=0.99,
            polygon=[(166, 56), (477, 56), (477, 101), (166, 101)],
            direction="horizontal",
        ),
        TextRegion(
            id=3,
            text="同じようなポーズをしていて",
            confidence=0.99,
            polygon=[(467, 80), (714, 80), (714, 101), (467, 101)],
            direction="horizontal",
        ),
        TextRegion(
            id=4,
            text="も、片方をSDキャラで描くことでメリハリとなる。",
            confidence=0.99,
            polygon=[(75, 104), (631, 104), (631, 155), (75, 155)],
            direction="horizontal",
        ),
    ]
    pipeline = make_pipeline(FakeOCR(regions), translator=RecordingTranslator())
    options = PipelineOptions(source_lang="ja", target_lang="zh", mode="auto")
    result = asyncio.run(pipeline.run(sample_png, options))

    assert len(result.blocks) == 1
    assert result.blocks[0].bbox == [75, 56, 639, 99]
    # Chrome keeps its original pixels; only the merged caption is redrawn.
    assert result.rendered.crop((72, 20, 300, 44)).tobytes() == (
        result.original.crop((72, 20, 300, 44)).tobytes()
    )


def test_pipeline_merges_wrapped_lines_before_translation(sample_png):
    regions = [
        TextRegion(
            id=0,
            text="The quick brown fox jumps over",
            confidence=0.99,
            polygon=[(50, 40), (550, 40), (550, 62), (50, 62)],
            direction="horizontal",
        ),
        TextRegion(
            id=1,
            text="the lazy dog.",
            confidence=0.98,
            polygon=[(50, 66), (350, 66), (350, 88), (50, 88)],
            direction="horizontal",
        ),
    ]
    translator = RecordingTranslator()
    pipeline = make_pipeline(FakeOCR(regions), translator=translator)
    options = PipelineOptions(source_lang="en", target_lang="zh", mode="auto")
    result = asyncio.run(pipeline.run(sample_png, options))
    assert len(result.regions) == 1
    assert result.regions[0].text == "The quick brown fox jumps over the lazy dog."
    assert result.regions[0].source_line_height == 22.0
    assert translator.seen is not None
    assert len(translator.seen) == 1
    assert translator.seen[0].text == result.regions[0].text


def test_pipeline_keeps_metadata_rows_out_of_translation(sample_png):
    regions = [
        TextRegion(
            id=0,
            text="ai_cosmic_horror_art4小时",
            confidence=0.99,
            polygon=[(50, 30), (330, 30), (330, 58), (50, 58)],
            direction="horizontal",
        ),
        TextRegion(
            id=1,
            text="In The Court of Yellow.",
            confidence=0.98,
            polygon=[(50, 70), (330, 70), (330, 100), (50, 100)],
            direction="horizontal",
        ),
    ]
    translator = RecordingTranslator()
    pipeline = make_pipeline(FakeOCR(regions), translator=translator)
    options = PipelineOptions(source_lang="en", target_lang="zh", mode="auto")
    result = asyncio.run(pipeline.run(sample_png, options))

    assert translator.seen is not None
    assert [region.text for region in translator.seen] == [
        "In The Court of Yellow."
    ]
    assert result.translations[0] == "ai_cosmic_horror_art 4小时"
    assert result.translations[1] == "EN In The Court of Yellow."
    assert result.regions[0].role == "metadata"
    # Chrome is never erased or redrawn: only the caption has a render block.
    assert len(result.blocks) == 1
    assert result.blocks[0].id == 1
    assert result.blocks[0].role == "text"
    metadata_box = (50, 30, 330, 58)
    assert (
        result.rendered.crop(metadata_box).tobytes()
        == result.original.crop(metadata_box).tobytes()
    )


def test_display_metadata_restores_dropped_timestamp_gap():
    assert _display_metadata("ai_cosmic_horror_art4小时") == (
        "ai_cosmic_horror_art 4小时"
    )
    # PP-OCR reads 小时 as a Cyrillic capital Es plus a modifier after a handle.
    assert _display_metadata(
        "ai_cosmic_horror_art4\u0421\u02b1"
    ) == "ai_cosmic_horror_art 4小时"
    assert _display_metadata(
        "ai_cosmic_horror_art4\u0421\u02b1", "en"
    ) == "ai_cosmic_horror_art 4\u0421\u02b1"


def test_pipeline_empty_ocr_returns_original(sample_png, options):
    pipeline = make_pipeline(FakeOCR([]))
    result = asyncio.run(pipeline.run(sample_png, options))
    assert result.empty
    assert result.regions == []
    assert result.blocks == []
    assert result.rendered.tobytes() == result.original.tobytes()
    assert any("No text" in warning for warning in result.warnings)


def test_pipeline_translation_failure_degrades(sample_png, sample_regions, options):
    pipeline = make_pipeline(
        FakeOCR(sample_regions), translator=FailingTranslator()
    )
    result = asyncio.run(pipeline.run(sample_png, options))
    assert result.degraded
    assert result.translation_provider == "fallback"
    assert result.translations[0] == sample_regions[0].text
    assert any("API exploded" in warning for warning in result.warnings)


def test_pipeline_translation_failure_can_be_strict(sample_png, sample_regions, options):
    pipeline = TranslationPipeline(
        ocr=FakeOCR(sample_regions),
        inpainter=OpenCVInpainter(),
        translator=FailingTranslator(),
        renderer=Renderer(FontBook()),
        translation_fallback_to_source=False,
    )
    with pytest.raises(TranslationError):
        asyncio.run(pipeline.run(sample_png, options))


def test_inpaint_and_translation_run_concurrently(sample_png, sample_regions, options):
    pipeline = make_pipeline(
        FakeOCR(sample_regions),
        translator=SlowTranslator(),
        inpainter=SlowInpainter(),
    )
    started = time.perf_counter()
    result = asyncio.run(pipeline.run(sample_png, options))
    elapsed = time.perf_counter() - started
    assert not result.empty
    # Sequential would be at least 0.6s; concurrent is ~0.3s + overhead.
    assert elapsed < 0.56


def test_pipeline_skips_translation_when_disabled(sample_png, sample_regions):
    options = PipelineOptions(source_lang="zh", target_lang="en", translate=False)
    pipeline = make_pipeline(FakeOCR(sample_regions), translator=RecordingTranslator())
    result = asyncio.run(pipeline.run(sample_png, options))
    assert result.translation_provider == "disabled"
    assert result.translations[0] == sample_regions[0].text


def test_pipeline_leaves_counter_pixels_untouched(sample_png):
    regions = [
        TextRegion(
            id=0,
            text="Hello world",
            confidence=0.99,
            polygon=[(72, 82), (470, 82), (470, 140), (72, 140)],
            direction="horizontal",
        ),
        TextRegion(
            id=1,
            text="390",
            confidence=0.99,
            polygon=[(574, 124), (690, 124), (690, 162), (574, 162)],
            direction="horizontal",
        ),
    ]
    translator = RecordingTranslator()
    pipeline = make_pipeline(FakeOCR(regions), translator=translator)
    options = PipelineOptions(source_lang="en", target_lang="zh", mode="auto")
    result = asyncio.run(pipeline.run(sample_png, options))

    # Counters are not sent to the model, are not masked, and are not redrawn.
    assert translator.seen is not None
    assert [region.text for region in translator.seen] == ["Hello world"]
    assert result.translations[1] == "390"
    assert [block.id for block in result.blocks] == [0]
    counter_box = (574, 124, 690, 162)
    assert (
        result.rendered.crop(counter_box).tobytes()
        == result.original.crop(counter_box).tobytes()
    )
    assert (
        result.clean.crop(counter_box).tobytes()
        == result.original.crop(counter_box).tobytes()
    )


def test_pipeline_keeps_misread_icon_pixels(sample_png):
    regions = [
        TextRegion(
            id=0,
            text="Hello world",
            confidence=0.99,
            polygon=[(72, 82), (470, 82), (470, 140), (72, 140)],
            direction="horizontal",
        ),
        # The like heart read as a low-confidence "O" beside the count.
        TextRegion(
            id=1,
            text="O",
            confidence=0.52,
            polygon=[(574, 300), (593, 300), (593, 317), (574, 317)],
            direction="horizontal",
        ),
        TextRegion(
            id=2,
            text="11万",
            confidence=0.99,
            polygon=[(593, 300), (623, 300), (623, 317), (593, 317)],
            direction="horizontal",
        ),
    ]
    translator = RecordingTranslator()
    pipeline = make_pipeline(FakeOCR(regions), translator=translator)
    options = PipelineOptions(source_lang="en", target_lang="zh", mode="auto")
    result = asyncio.run(pipeline.run(sample_png, options))

    assert translator.seen is not None
    assert [region.text for region in translator.seen] == ["Hello world"]
    assert result.translations[1] == "O"
    assert result.translations[2] == "11万"
    assert [block.id for block in result.blocks] == [0]
    icon_box = (574, 300, 623, 317)
    assert (
        result.rendered.crop(icon_box).tobytes()
        == result.original.crop(icon_box).tobytes()
    )
    assert (
        result.clean.crop(icon_box).tobytes()
        == result.original.crop(icon_box).tobytes()
    )


def test_pipeline_rejects_bad_image(options):
    pipeline = make_pipeline(FakeOCR([]))
    with pytest.raises(InvalidImageError):
        asyncio.run(pipeline.run(b"not an image", options))
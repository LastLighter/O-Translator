from __future__ import annotations

import asyncio

from core_pipeline.inpainting import FastInpaint, OpenCVInpainter
from core_pipeline.pipeline import TranslationPipeline
from core_pipeline.types import InpaintOutput, OcrOutput, TranslationOutput
from core_pipeline.typesetting import FontBook, Renderer


class FakeOCR:
    name = "fake-ocr"

    def __init__(self, regions):
        self.regions = regions

    def read(self, image):
        return OcrOutput(regions=self.regions, engine=self.name, elapsed_ms=1.0)


class RecordingTranslator:
    name = "recording"
    model = "test"

    async def translate(self, regions, options):
        return TranslationOutput(
            translations={region.id: f"EN {region.text}" for region in regions},
            provider=self.name,
            model=self.model,
            elapsed_ms=1.0,
        )

    async def aclose(self):
        return None


class ProgressiveInpainter(OpenCVInpainter):
    """OpenCV does the work, but advertises the progressive fast-pass API."""

    def fast_pass(self, image, mask, regions=None):
        output = self.inpaint(image, mask)
        return FastInpaint(
            image=output.image,
            plans=[],
            elapsed_ms=output.elapsed_ms,
            engine=output.engine,
        )

    def inpaint_with_fast(self, image, mask, regions, fast):
        return InpaintOutput(
            image=fast.image,
            engine=fast.engine,
            elapsed_ms=fast.elapsed_ms,
        )


def test_preview_callback_runs_before_final_result(sample_png, sample_regions):
    from core_pipeline.types import PipelineOptions

    previews: list[tuple[int, int, int]] = []
    pipeline = TranslationPipeline(
        ocr=FakeOCR(sample_regions),
        inpainter=ProgressiveInpainter(),
        translator=RecordingTranslator(),
        renderer=Renderer(FontBook()),
        progressive_preview=True,
    )

    async def on_preview(image, blocks, regions, translations, timings):
        previews.append((image.width, image.height, len(blocks)))
        assert translations
        assert "preview_ready_ms" in timings

    result = asyncio.run(
        pipeline.run(
            sample_png,
            PipelineOptions(source_lang="zh", target_lang="en", mode="game"),
            on_preview=on_preview,
        )
    )
    assert previews == [(760, 460, 2)]
    assert not result.empty
    assert len(result.blocks) == 2
    assert "preview_ready_ms" in result.timings_ms
    assert "inpaint_fast_ms" in result.timings_ms

from __future__ import annotations

import asyncio
import importlib.util
import io
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from core_pipeline.inpainting import OpenCVInpainter
from core_pipeline.ocr import RapidOCRBackend
from core_pipeline.pipeline import TranslationPipeline
from core_pipeline.translation import EchoTranslator
from core_pipeline.types import PipelineOptions
from core_pipeline.typesetting import FontBook, Renderer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OTR_RUN_INTEGRATION") != "1",
        reason="set OTR_RUN_INTEGRATION=1 to load real OCR models",
    ),
    pytest.mark.skipif(
        not importlib.util.find_spec("rapidocr"),
        reason="rapidocr is not installed",
    ),
]


def _image() -> Image.Image:
    image = Image.new("RGB", (900, 320), (248, 247, 242))
    draw = ImageDraw.Draw(image)
    font_path = next(
        (
            path
            for path in (
                "C:/Windows/Fonts/arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            )
            if Path(path).is_file()
        ),
        None,
    )
    font = (
        ImageFont.truetype(font_path, 54)
        if font_path
        else ImageFont.load_default(size=54)
    )
    draw.text((50, 60), "Hello O-Translator", font=font, fill=(18, 18, 18))
    draw.text((50, 165), "Real OCR pipeline check", font=font, fill=(18, 18, 18))
    return image


def test_real_rapidocr_pipeline(tmp_path):
    image = _image()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    pipeline = TranslationPipeline(
        ocr=RapidOCRBackend(lang="en", min_confidence=0.3, device="cpu"),
        inpainter=OpenCVInpainter(),
        translator=EchoTranslator(),
        renderer=Renderer(FontBook()),
    )
    result = asyncio.run(
        pipeline.run(
            buffer.getvalue(),
            PipelineOptions(source_lang="en", target_lang="en", mode="ui"),
        )
    )
    assert not result.empty
    assert len(result.regions) >= 1
    joined = " ".join(region.text for region in result.regions).lower()
    assert "translator" in joined or "hello" in joined
    result.rendered.save(tmp_path / "rendered.png")
    assert (tmp_path / "rendered.png").is_file()
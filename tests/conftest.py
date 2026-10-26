from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from core_pipeline.types import PipelineOptions, TextRegion


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


@pytest.fixture
def sample_image() -> Image.Image:
    image = Image.new("RGB", (760, 460), (248, 246, 239))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((46, 58, 496, 156), radius=16, fill=(24, 29, 38))
    draw.text(
        (72, 82),
        "你好，欢迎使用 O-Translator！",
        font=_font(30),
        fill=(245, 243, 236),
    )
    draw.ellipse((540, 62, 716, 238), fill=(103, 86, 154))
    draw.text((574, 124), "测试文本", font=_font(26), fill=(255, 255, 255))
    return image


@pytest.fixture
def sample_png(sample_image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    sample_image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_regions() -> list[TextRegion]:
    return [
        TextRegion(
            id=0,
            text="你好，欢迎使用 O-Translator！",
            confidence=0.99,
            polygon=[(72, 82), (470, 82), (470, 140), (72, 140)],
            direction="horizontal",
        ),
        TextRegion(
            id=1,
            text="测试文本",
            confidence=0.97,
            polygon=[(574, 124), (690, 124), (690, 162), (574, 162)],
            direction="horizontal",
        ),
    ]


@pytest.fixture
def options() -> PipelineOptions:
    return PipelineOptions(source_lang="zh", target_lang="en", mode="game")

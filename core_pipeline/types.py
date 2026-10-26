"""Typed domain models and pipeline contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field, field_validator

TextDirection = Literal["horizontal", "vertical"]
TextRole = Literal["text", "metadata", "hashtag", "keep"]
ScenarioMode = Literal["auto", "manga", "game", "novel", "ui"]
ReadingOrder = Literal["auto", "ltr", "rtl"]


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0

    @classmethod
    def from_polygon(cls, polygon: list[tuple[float, float]]) -> BBox:
        if not polygon:
            return cls(x=0, y=0, w=0, h=0)
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        x0, y0 = max(0, int(math.floor(min(xs)))), max(0, int(math.floor(min(ys))))
        x1, y1 = int(math.ceil(max(xs))), int(math.ceil(max(ys)))
        return cls(x=x0, y=y0, w=max(0, x1 - x0), h=max(0, y1 - y0))

    def clamp(self, width: int, height: int) -> BBox:
        x0 = min(max(0, self.x), max(0, width - 1))
        y0 = min(max(0, self.y), max(0, height - 1))
        x1 = min(max(x0 + 1, self.right), width)
        y1 = min(max(y0 + 1, self.bottom), height)
        return BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)

    def expanded(self, px: int, width: int, height: int) -> BBox:
        return BBox(
            x=self.x - px,
            y=self.y - px,
            w=self.w + px * 2,
            h=self.h + px * 2,
        ).clamp(width, height)


class TextRegion(BaseModel):
    """One OCR line. ``polygon`` is kept for mask generation."""

    id: int = 0
    text: str
    confidence: float = 1.0
    polygon: list[tuple[float, float]] = Field(default_factory=list)
    direction: TextDirection = "horizontal"
    angle: float = 0.0
    # Social/UI screenshots mix the post body with chrome that must not be
    # welded into it. ``metadata`` covers usernames, relative timestamps and
    # action labels; ``hashtag`` covers tag-only lines that keep their own
    # small, regular-weight block; ``keep`` covers language-neutral counters
    # (``18``, ``8.2万``, ``1.2K views``) whose original pixels must survive
    # untouched. ``metadata`` and ``keep`` are never masked or redrawn.
    role: TextRole = "text"
    # Optional layout box when several OCR pieces were merged for translation
    # (the title's large glyphs anchor the redraw, not the union box).
    render_bbox: list[int] | None = None
    # Extra mask growth for merged caption ribbons whose first line the
    # detector missed; the leftover strokes would otherwise ghost through.
    mask_expand_px: int = 0
    # Effective height of one original source line when several OCR lines were
    # merged (a stacked caption contributes its own median, not the union-box
    # height). ``capacity_hint`` uses it to keep the LLM character budget tied
    # to the source line height, and the renderer caps the translated font at
    # this size so a tall paragraph box never inflates a short translation.
    source_line_height: float = 0.0

    @property
    def bbox(self) -> BBox:
        return BBox.from_polygon(self.polygon)

    def capacity_hint(self, target_lang: str) -> int:
        """Rough number of target characters that fit, for the LLM prompt."""
        if self.role == "hashtag":
            # A tag row is a list, not prose: dropping tags to meet the box's
            # single-line budget would silently delete metadata. Keep a
            # generous floor so every tag survives translation.
            return max(96, len(self.text) + 40)
        box = self.bbox
        if box.w <= 0 or box.h <= 0:
            return max(4, len(self.text))
        cjk_target = target_lang.lower() in {"zh", "ja", "ko", "chinese", "japanese", "korean"}
        width_factor = 1.0 if cjk_target else 0.56
        # Maximum font that could fit one character, then estimate wraps.
        font = max(7.0, min(box.h * 0.82, box.w * 0.85))
        if self.source_line_height > 0:
            # A merged paragraph's union box is several lines tall. Scaling the
            # font from that height would tell the model a 5-line post fits in
            # one giant line and make it over-compress the translation.
            font = max(7.0, min(font, self.source_line_height * 0.82))
        per_line = max(1, int(box.w / (font * width_factor)))
        lines = max(1, int(round(box.h / (font * 1.18))))
        return max(4, min(500, per_line * lines))


class PipelineOptions(BaseModel):
    source_lang: str = "auto"
    target_lang: str = "zh"
    mode: ScenarioMode = "auto"
    reading_order: ReadingOrder = "auto"
    dilate_px: int | None = None
    font_scale: float = 1.0
    translate: bool = True

    @field_validator("source_lang", "target_lang")
    @classmethod
    def normalize_lang(cls, value: str) -> str:
        value = value.strip().lower()
        aliases = {
            "cn": "zh",
            "zh-cn": "zh",
            "zh-tw": "zh",
            "jp": "ja",
            "jpn": "ja",
            "kr": "ko",
            "kor": "ko",
            "eng": "en",
            "auto-detect": "auto",
        }
        return aliases.get(value, value or "auto")

    @field_validator("font_scale")
    @classmethod
    def validate_scale(cls, value: float) -> float:
        return min(1.6, max(0.6, value))

    @field_validator("dilate_px")
    @classmethod
    def validate_dilate(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return min(7, max(3, value))


class TranslationBatch(BaseModel):
    items: list[dict[str, object]]


@dataclass(slots=True)
class OcrOutput:
    regions: list[TextRegion]
    engine: str
    elapsed_ms: float
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InpaintOutput:
    image: Image.Image
    engine: str
    elapsed_ms: float
    warnings: list[str] = field(default_factory=list)
    # Region-level telemetry for the progressive/preview pipeline. Defaults keep
    # the object source-compatible with simple OpenCV-only callers.
    fast_regions: int = 0
    lama_regions: int = 0
    roi_patches: int = 0
    strategy: str = ""


@dataclass(slots=True)
class TranslationOutput:
    translations: dict[int, str]
    provider: str
    model: str
    elapsed_ms: float
    degraded: bool = False
    warnings: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class RenderBlock(BaseModel):
    id: int
    source_text: str
    translated_text: str
    bbox: list[int]
    polygon: list[tuple[float, float]]
    direction: TextDirection
    role: TextRole = "text"
    align: Literal["left", "center", "right"] = "center"
    layout: Literal["horizontal", "vertical"]
    font_size: int
    line_height: int
    lines: list[str]
    text_color: list[int]
    background_color: list[int]
    overflow: bool = False
    fit_ratio: float = 1.0
    font_weight: str = "regular"
    stroke_color: list[int] | None = None
    stroke_width: int = 0


@dataclass(slots=True)
class RenderOutput:
    image: Image.Image
    blocks: list[RenderBlock]
    warnings: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


@dataclass(slots=True)
class PipelineResult:
    job_id: str
    original: Image.Image
    clean: Image.Image
    mask: Image.Image
    rendered: Image.Image
    regions: list[TextRegion]
    translations: dict[int, str]
    blocks: list[RenderBlock]
    ocr_engine: str
    inpaint_engine: str
    translation_provider: str
    translation_model: str
    degraded: bool
    warnings: list[str]
    timings_ms: dict[str, float]
    options: PipelineOptions
    source_size: tuple[int, int]
    processed_size: tuple[int, int]
    translation_usage: dict[str, int] = field(default_factory=dict)
    inpaint_strategy: str = ""
    empty: bool = False
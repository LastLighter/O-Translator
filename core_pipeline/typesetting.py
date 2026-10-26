"""Font selection, wrapping, auto-fit, and color analysis for typesetting."""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core_pipeline.types import BBox, RenderBlock, RenderOutput, TextRegion

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CJK_TARGETS = {"zh", "ja", "ko", "zh-tw", "chinese", "japanese", "korean"}
_NO_LINE_START = set("、。，．！？：；）」』】〕》〉”’ー～…,.!?;:)]}%")
_NO_LINE_END = set("（「『【〔《〈“‘([{")
_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:[’'\-][A-Za-z0-9]+)*|[^\S\r\n]+|\n|.",
    re.UNICODE,
)

# Font families used for manga typesetting. Display faces are deliberately
# optional: installations without them fall back to the bold face.
_FONT_FAMILIES: dict[str, list[str]] = {
    "regular": [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/NotoSansSC-VF.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ],
    "bold": [
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/Dengb.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ],
    "display": [
        "C:/Windows/Fonts/STHUPO.TTF",
        "C:/Windows/Fonts/STXINGKA.TTF",
        "C:/Windows/Fonts/STCAIYUN.TTF",
        "C:/Windows/Fonts/FZSTK.TTF",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ],
    "serif": [
        "C:/Windows/Fonts/NotoSerifSC-VF.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ],
}


class FontBook:
    """Locate multilingual font families once, then cache instances by size."""

    def __init__(self, font_path: str | None = None) -> None:
        self.paths: dict[str, str | None] = {}
        regular = font_path if font_path and Path(font_path).is_file() else None
        if regular:
            self.paths["regular"] = regular
        for family, candidates in _FONT_FAMILIES.items():
            if family in self.paths:
                continue
            self.paths[family] = self._resolve(candidates)
        for folder in (PROJECT_ROOT / "models" / "fonts",):
            if folder.is_dir():
                for pattern in ("*.ttf", "*.otf", "*.ttc"):
                    matches = sorted(folder.glob(pattern))
                    if matches and not self.paths.get("regular"):
                        self.paths["regular"] = str(matches[0])
                        break
        self.path = self.paths.get("regular")
        self.warning = (
            None
            if self.path
            else (
                "No CJK-capable font found. Install Microsoft YaHei, Noto Sans CJK, "
                "or set OTR_FONT_PATH to a TTF/TTC file."
            )
        )
        self._cache: dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    @staticmethod
    def _resolve(candidates: list[str]) -> str | None:
        for candidate in candidates:
            if Path(candidate).is_file():
                return candidate
        return None

    def resolve_weight(self, weight: str) -> str:
        if self.paths.get(weight):
            return weight
        if weight in {"display", "bold"} and self.paths.get("bold"):
            return "bold"
        if weight == "serif" and self.paths.get("regular"):
            return "regular"
        return "regular" if self.paths.get("regular") else weight

    def get(
        self, size: int, weight: str = "regular"
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        size = max(1, int(size))
        family = self.resolve_weight(weight)
        key = (family, size)
        if key in self._cache:
            return self._cache[key]
        path = self.paths.get(family) or self.path
        if path:
            try:
                font = ImageFont.truetype(path, size=size, index=0)
            except OSError:
                font = ImageFont.load_default()
        else:
            try:
                font = ImageFont.load_default(size=size)
            except TypeError:  # older Pillow
                font = ImageFont.load_default()
        self._cache[key] = font
        return font


@dataclass(slots=True)
class FitResult:
    lines: list[str]
    font_size: int
    line_height: int
    width: int
    height: int
    overflow: bool
    layout: str


def is_cjk_target(target_lang: str) -> bool:
    return target_lang.lower() in _CJK_TARGETS


def _line_height(
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    size: int,
    *,
    tight: bool = False,
) -> int:
    if tight:
        # CJK lettering is set on a near-square em box. Using the font's full
        # ascent+descent (~1.3 em for YaHei) leaves manga columns visibly loose
        # and wastes horizontal space.
        return max(1, int(round(size * 1.08)))
    try:
        ascent, descent = font.getmetrics()
        return max(size, int(math.ceil(ascent + descent)))
    except (AttributeError, OSError):
        return max(1, int(round(size * 1.2)))


@lru_cache(maxsize=32768)
def _cached_text_width(font, text: str) -> float:
    try:
        return float(font.getlength(text))
    except AttributeError:
        bbox = font.getbbox(text)
        return float(bbox[2] - bbox[0])


def _text_width(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> float:
    """Measure a string, memoized by (font instance, text).

    Fitting tries several sizes and re-measures the winning lines; repeated UI
    labels and identical captions also hit this cache across regions.
    """
    return _cached_text_width(font, text)


def _split_long_token(token: str, font, max_width: float) -> list[str]:
    pieces: list[str] = []
    current = ""
    for char in token:
        candidate = current + char
        if current and _text_width(font, candidate) > max_width:
            pieces.append(current)
            current = char
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces or [token]


_CJK_FAST_RANGES = (
    (0x2000, 0x206F),  # general punctuation (ellipsis, dashes, curly quotes)
    (0x3000, 0x30FF),  # CJK punctuation, hiragana, katakana
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),
    (0xF900, 0xFAFF),
    (0xFF00, 0xFFEF),  # full-width forms
)


def _is_cjk_fast_char(char: str) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in _CJK_FAST_RANGES)


@lru_cache(maxsize=8192)
def _is_cjk_fast_char_cached(char: str) -> bool:
    return _is_cjk_fast_char(char)


@lru_cache(maxsize=2048)
def _paragraph_is_fast_cjk(paragraph: str) -> bool:
    """True when every glyph is CJK and no whitespace complicates wrapping.

    Spaces are excluded so the fast path never has to reproduce the generic
    wrapper's "space unless the next token is CJK" separator rule.
    """
    if not paragraph:
        return False
    return all(
        not char.isspace() and _is_cjk_fast_char_cached(char)
        for char in paragraph
    )


def _wrap_cjk(
    paragraph: str, font, max_width: float
) -> tuple[list[str], bool]:
    """Fast greedy wrap for CJK-only text using per-glyph advances.

    CJK fonts have no kerning between these glyphs, so ``getlength(line)`` is
    the sum of its glyph advances. Measuring glyphs once (and caching them)
    avoids the O(n^2) prefix measurement that dominated font fitting. Returns
    ``(lines, verified)``; when a measured line disagrees with the additive
    width, the caller falls back to the exact wrapper.
    """
    advances: dict[str, float] = {}

    def advance(char: str) -> float:
        width = advances.get(char)
        if width is None:
            width = _text_width(font, char)
            advances[char] = width
        return width

    result: list[str] = []
    widths: list[float] = []
    line = ""
    line_width = 0.0
    for char in paragraph:
        if char.isspace():
            # Like the generic wrapper, a space between CJK glyphs disappears.
            continue
        if not line:
            line = char
            line_width = advance(char)
            continue
        candidate_width = line_width + advance(char)
        if candidate_width <= max_width:
            line += char
            line_width = candidate_width
            continue
        # Closing punctuation must not begin a new line.
        if char in _NO_LINE_START:
            line += char
            line_width = candidate_width
            continue
        # Opening punctuation should not dangle at the end of a line.
        if line[-1] in _NO_LINE_END and len(line) > 1:
            moved = line[-1]
            line = line[:-1]
            line_width -= advance(moved)
            if line:
                result.append(line)
                widths.append(line_width)
            line = moved + char
            line_width = advance(moved) + advance(char)
        else:
            result.append(line)
            widths.append(line_width)
            line = char
            line_width = advance(char)
    if line:
        result.append(line)
        widths.append(line_width)
    # CJK fonts normally have no kerning for these glyphs; when they do the
    # generic wrapper is used so the result stays pixel-exact.
    for measured, additive in zip(result, widths, strict=False):
        if abs(_text_width(font, measured) - additive) > 0.01:
            return result, False
    return result, True


def _wrap_generic(
    paragraph: str, font, max_width: float, *, allow_empty: bool
) -> list[str]:
    """Exact pixel-measured wrap for one paragraph (non-CJK fast path)."""
    lines: list[str] = []
    tokens = _TOKEN_RE.findall(paragraph)
    line = ""
    pending_space = False
    for token in tokens:
        if token.isspace() and "\n" not in token:
            pending_space = bool(line)
            continue
        if token == "\n":
            if line:
                lines.append(line.rstrip())
                line = ""
            lines.append("")
            pending_space = False
            continue

        separator = " " if pending_space and line and not _is_cjk_char(token[0]) else ""
        candidate = line + separator + token
        width = _text_width(font, candidate.rstrip())
        if width <= max_width or not line:
            if width > max_width and not line and len(token) > 1:
                pieces = _split_long_token(token, font, max_width)
                lines.extend(pieces[:-1])
                line = pieces[-1]
            else:
                line = candidate
            pending_space = False
            continue

        # Closing punctuation must not begin a new line.
        if token and token[0] in _NO_LINE_START:
            line += token
            pending_space = False
            continue
        # Opening punctuation should not dangle at the end of a line.
        if line and line[-1] in _NO_LINE_END:
            moved = line[-1]
            line = line[:-1].rstrip()
            if line:
                lines.append(line)
            line = moved + token
        else:
            lines.append(line.rstrip())
            line = token.lstrip()
            if len(line) > 1 and _text_width(font, line) > max_width:
                pieces = _split_long_token(line, font, max_width)
                lines.extend(pieces[:-1])
                line = pieces[-1]
        pending_space = False
    if line or allow_empty:
        lines.append(line.rstrip())
    return lines


def wrap_text(text: str, font, max_width: float) -> list[str]:
    """Pixel-aware wrapping for CJK and space-delimited scripts."""
    max_width = max(1.0, max_width)
    result: list[str] = []
    paragraphs = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for paragraph in paragraphs:
        if paragraph == "":
            result.append("")
            continue
        if not paragraph.isspace() and _paragraph_is_fast_cjk(paragraph):
            fast_lines, verified = _wrap_cjk(paragraph, font, max_width)
            if verified:
                result.extend(fast_lines or [paragraph])
                continue
        result.extend(
            _wrap_generic(paragraph, font, max_width, allow_empty=not result)
        )
    # Remove duplicated blank lines introduced by explicit empty paragraphs.
    cleaned: list[str] = []
    for line in result:
        if line == "" and cleaned and cleaned[-1] == "":
            continue
        cleaned.append(line)
    return cleaned or [text]


def _fit_horizontal(
    text: str,
    box: BBox,
    fontbook: FontBook,
    *,
    min_size: int,
    font_scale: float,
    padding: int,
    weight: str = "regular",
    tight: bool = False,
    max_size: int | None = None,
) -> FitResult:
    max_w = max(2.0, box.w - padding * 2)
    max_h = max(2.0, box.h - padding * 2)
    start = int(min(max_w, max_h) * 0.98 * font_scale)
    if max_size is not None:
        start = min(start, max_size)
    start = max(min_size, min(start, 220))
    best: FitResult | None = None
    for size in range(start, min_size - 1, -1):
        font = fontbook.get(size, weight)
        lines = wrap_text(text, font, max_w)
        line_h = _line_height(font, size, tight=tight)
        total_h = max(1, len(lines)) * line_h
        measured_w = max((_text_width(font, line) for line in lines), default=1.0)
        overflow = measured_w > max_w + 0.5 or total_h > max_h + 0.5
        result = FitResult(
            lines=lines,
            font_size=size,
            line_height=line_h,
            width=int(math.ceil(measured_w)),
            height=int(math.ceil(total_h)),
            overflow=overflow,
            layout="horizontal",
        )
        if not overflow:
            return result
        best = result
    if best is None:  # pragma: no cover - start is always >= min_size
        font = fontbook.get(min_size, weight)
        lines = wrap_text(text, font, max_w)
        line_h = _line_height(font, min_size, tight=tight)
        best = FitResult(
            lines=lines,
            font_size=min_size,
            line_height=line_h,
            width=int(max((_text_width(font, line) for line in lines), default=1)),
            height=len(lines) * line_h,
            overflow=True,
            layout="horizontal",
        )
    return best


def _split_vertical(text: str, per_column: int) -> list[str]:
    """Split a vertical sentence into columns with basic kinsoku rules."""
    per_column = max(1, per_column)
    columns: list[str] = []
    current = ""
    for char in text:
        if len(current) >= per_column:
            # Closing punctuation must not open a new column.
            if char in _NO_LINE_START and current:
                current += char
                continue
            # Opening punctuation must not dangle at a column end.
            if current and current[-1] in _NO_LINE_END and len(current) > 1:
                moved = current[-1]
                current = current[:-1]
                columns.append(current)
                current = moved + char
                continue
            columns.append(current)
            current = char
        else:
            current += char
    if current:
        columns.append(current)
    return columns or [text]


def _fit_vertical(
    text: str,
    box: BBox,
    fontbook: FontBook,
    *,
    min_size: int,
    font_scale: float,
    padding: int,
    weight: str = "regular",
    tight: bool = True,
    max_size: int | None = None,
) -> FitResult:
    max_w = max(2.0, box.w - padding * 2)
    max_h = max(2.0, box.h - padding * 2)
    clean = text.replace("\r", "").replace("\n", "")
    if not clean:
        clean = text
    start = int(min(max_w, max_h) * 0.98 * font_scale)
    if max_size is not None:
        start = min(start, max_size)
    start = max(min_size, min(start, 220))
    best: FitResult | None = None
    for size in range(start, min_size - 1, -1):
        font = fontbook.get(size, weight)
        line_h = _line_height(font, size, tight=tight)
        max_cols = max(0, int(max_w // max(1, line_h)))
        per_col = max(1, int(max_h // max(1, line_h)))
        columns = _split_vertical(clean, per_col)
        total_w = len(columns) * line_h
        measured_h = max((len(column) for column in columns), default=0) * line_h
        overflow = (
            max_cols < 1
            or len(columns) > max_cols
            or measured_h > max_h + 0.5
        )
        result = FitResult(
            lines=columns,
            font_size=size,
            line_height=line_h,
            width=int(math.ceil(total_w)),
            height=int(math.ceil(measured_h)),
            overflow=overflow,
            layout="vertical",
        )
        if not overflow:
            return result
        best = result
    if best is None:  # pragma: no cover - start is always >= min_size
        font = fontbook.get(min_size, weight)
        line_h = _line_height(font, min_size, tight=tight)
        per_col = max(1, int(max_h // max(1, line_h)))
        columns = _split_vertical(clean, per_col)
        best = FitResult(
            lines=columns,
            font_size=min_size,
            line_height=line_h,
            width=len(columns) * line_h,
            height=max((len(column) for column in columns), default=0) * line_h,
            overflow=True,
            layout="vertical",
        )
    return best


def fit_text(
    text: str,
    box: BBox,
    fontbook: FontBook,
    *,
    target_lang: str = "zh",
    source_direction: str = "horizontal",
    min_size: int = 7,
    font_scale: float = 1.0,
    weight: str = "regular",
    max_size: int | None = None,
    tight: bool | None = None,
) -> FitResult:
    """Choose font size/layout so text fits the original OCR box.

    ``tight`` defaults to the CJK near-square line box used for manga/UI
    lettering. Reconstructed paragraphs pass ``False`` so their line spacing
    follows normal font metrics instead of being compressed like a speech
    bubble.
    """
    if min(box.w, box.h) < 14:
        padding = 0
    elif min(box.w, box.h) < 20:
        padding = 1
    else:
        padding = 2
    tight = is_cjk_target(target_lang) if tight is None else bool(tight)
    vertical = (
        source_direction == "vertical"
        and is_cjk_target(target_lang)
        and any(_is_cjk_char(char) for char in text)
    )
    if vertical:
        return _fit_vertical(
            text,
            box,
            fontbook,
            min_size=min_size,
            font_scale=font_scale,
            padding=padding,
            weight=weight,
            tight=tight,
            max_size=max_size,
        )
    return _fit_horizontal(
        text,
        box,
        fontbook,
        min_size=min_size,
        font_scale=font_scale,
        padding=padding,
        weight=weight,
        tight=tight,
        max_size=max_size,
    )


# ---------------------------------------------------------------------------
# Color analysis
# ---------------------------------------------------------------------------
def relative_luminance(color: tuple[int, int, int] | list[int]) -> float:
    channels = []
    for value in color[:3]:
        c = max(0.0, min(255.0, float(value))) / 255.0
        channels.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def contrast_ratio(
    first: tuple[int, int, int] | list[int],
    second: tuple[int, int, int] | list[int],
) -> float:
    l1, l2 = relative_luminance(first), relative_luminance(second)
    light, dark = max(l1, l2), min(l1, l2)
    return (light + 0.05) / (dark + 0.05)


def _dominant_color(pixels: np.ndarray) -> tuple[int, int, int]:
    if len(pixels) == 0:
        return (0, 0, 0)
    quantized = (pixels // 16).astype(np.int16)
    codes = quantized[:, 0] * 256 + quantized[:, 1] * 16 + quantized[:, 2]
    values, counts = np.unique(codes, return_counts=True)
    code = int(values[int(np.argmax(counts))])
    target = np.array([(code // 256), (code // 16) % 16, code % 16], dtype=np.float32)
    quantized_float = pixels.astype(np.float32) // 16
    mask = np.all(quantized_float == target, axis=1)
    selected = pixels[mask] if np.any(mask) else pixels
    return tuple(int(v) for v in np.median(selected, axis=0)[:3])


def extract_colors(
    image: Image.Image,
    box: BBox,
    *,
    pixels: np.ndarray | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Estimate dominant foreground text and background colors in a box.

    ``pixels`` is an optional cached RGB array of the whole page. Converting
    the page once per render instead of once per box removes a large amount of
    array copying on text-heavy pages.
    """
    box = box.clamp(*image.size)
    if pixels is None:
        pixels = np.asarray(image.convert("RGB"))
    crop = pixels[box.y : box.bottom, box.x : box.right]
    if crop.size == 0:
        return (0, 0, 0), (255, 255, 255)
    pixels = crop.reshape(-1, 3)
    border_parts = [
        crop[0, :, :],
        crop[-1, :, :],
        crop[:, 0, :],
        crop[:, -1, :],
    ]
    border = np.concatenate([part.reshape(-1, 3) for part in border_parts], axis=0)
    background = _dominant_color(border if len(border) >= 8 else pixels)

    distances = np.linalg.norm(pixels.astype(np.float32) - np.array(background), axis=1)
    threshold = max(42.0, float(np.percentile(distances, 72)))
    candidates = pixels[distances >= threshold]
    if len(candidates) < max(3, len(pixels) // 100):
        foreground = (0, 0, 0) if relative_luminance(background) > 0.5 else (255, 255, 255)
    else:
        # The strongest-contrast quartile contains glyph cores and suppresses
        # colored artwork that merely happens to sit inside the box.
        candidate_distances = distances[distances >= threshold]
        if len(candidate_distances) > 8:
            strong_cutoff = float(np.percentile(candidate_distances, 65))
            strong = pixels[distances >= strong_cutoff]
            foreground = _dominant_color(strong if len(strong) >= 4 else candidates)
        else:
            foreground = _dominant_color(candidates)

    if contrast_ratio(foreground, background) < 3.2:
        foreground = (0, 0, 0) if relative_luminance(background) > 0.5 else (255, 255, 255)
    return foreground, background


def analyze_text_style(
    image: Image.Image,
    box: BBox,
    *,
    mode: str = "auto",
    font_size: int = 16,
    pixels: np.ndarray | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int] | None, int, tuple[int, int, int]]:
    """Choose fill, outline, outline width, and background for rendered text.

    Manga lettering is almost always white with a dark outline over artwork. The
    previous "dominant contrast" heuristic latched onto speed lines and painted
    black text on dark art. Looking at the box's median luminance distinguishes
    artwork panels (use white + outline) from bright paper/caption areas (keep
    the sampled ink color plus a light outline).

    ``pixels`` is the cached RGB page array, shared with :func:`extract_colors`
    so a render only converts the source image once.
    """
    box = box.clamp(*image.size)
    if pixels is None:
        pixels = np.asarray(image.convert("RGB"))
    crop = pixels[box.y : box.bottom, box.x : box.right]
    if crop.size == 0:
        return (0, 0, 0), None, 0, (255, 255, 255)
    pixels_f = crop.reshape(-1, 3).astype(np.float32) / 255.0
    linear = np.where(
        pixels_f <= 0.04045, pixels_f / 12.92, ((pixels_f + 0.055) / 1.055) ** 2.4
    )
    luminance = (
        0.2126 * linear[:, 0] + 0.7152 * linear[:, 1] + 0.0722 * linear[:, 2]
    )
    median_luminance = float(np.median(luminance))
    foreground, background = extract_colors(image, box, pixels=pixels)

    if mode in {"manga", "novel", "auto"}:
        if median_luminance < 0.60:
            # Dark/busy artwork: white lettering with a dark outline, exactly
            # like the source manga.
            outline = (18, 16, 22)
            width = max(1, min(3, int(round(font_size / 18))))
            return (255, 255, 255), outline, width, background
        if relative_luminance(foreground) > 0.72:
            foreground = (20, 20, 20)
        width = 1 if font_size >= 14 else 0
        return foreground, (255, 255, 255) if width else None, width, background
    return foreground, None, 0, background


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
class Renderer:
    def __init__(
        self,
        fontbook: FontBook,
        *,
        min_font_size: int = 7,
        font_scale: float = 1.0,
    ) -> None:
        self.fontbook = fontbook
        self.min_font_size = max(5, min_font_size)
        self.font_scale = max(0.6, min(1.6, font_scale))

    def render(
        self,
        original: Image.Image,
        clean: Image.Image,
        regions: list[TextRegion],
        translations: dict[int, str],
        *,
        target_lang: str = "zh",
        font_scale: float | None = None,
        mode: str = "auto",
    ) -> RenderOutput:
        result = clean.copy().convert("RGB")
        draw = ImageDraw.Draw(result)
        # Color analysis reads from the untouched source page. Convert it once
        # for the whole render: doing this per box copied a full-page RGB array
        # twice per region (2x N full image allocations on text-heavy pages).
        source_pixels = np.asarray(original.convert("RGB"))
        blocks: list[RenderBlock] = []
        warnings: list[str] = []
        scale = self.font_scale if font_scale is None else max(0.6, min(1.6, font_scale))
        # OCR box heights vary by a few pixels around the same type size. Use
        # the page's typical line height as a gentle cap so two identical-looking
        # captions are not set at 18 px and 25 px just because one box caught a
        # descender or an extra pixel of padding. Clearly larger display lines
        # are exempt and may still fill their own box.
        horizontal_heights = [
            region.bbox.h
            for region in regions
            if region.direction == "horizontal"
            and region.bbox.h > 0
            and region.role not in {"metadata", "keep"}
        ]
        page_line_height = (
            float(statistics.median(horizontal_heights))
            if horizontal_heights
            else 0.0
        )

        for region in regions:
            # Chrome is excluded from the mask, so its original pixels are
            # still on the canvas. Redrawing it would only paint translations
            # over avatars, verified badges and action icons.
            if region.role in {"metadata", "keep"}:
                continue
            box = region.bbox.clamp(*result.size)
            if region.render_bbox and len(region.render_bbox) == 4:
                candidate = BBox(
                    x=int(region.render_bbox[0]),
                    y=int(region.render_bbox[1]),
                    w=int(region.render_bbox[2]),
                    h=int(region.render_bbox[3]),
                ).clamp(*result.size)
                if candidate.w >= 3 and candidate.h >= 3:
                    box = candidate
            if box.w < 3 or box.h < 3:
                continue
            translated = (translations.get(region.id) or region.text).strip()
            if not translated:
                translated = region.text
            # A merged paragraph's union box is several source lines tall.
            # Sizing text from that union height lets a short translation
            # balloon to 3-5x the original type size (the "giant hashtag"
            # failure), so cap it at one source line. ``source_line_height``
            # is the weighted median of the physical OCR lines; using the
            # union-box height here is what set a 20 px Japanese post at
            # 27 px and made it overlap the next paragraph.
            paragraph = region.source_line_height > 0
            if region.direction == "vertical":
                max_size = max(self.min_font_size, int(round(box.w * 0.72)))
            else:
                max_size = max(self.min_font_size, int(round(box.h * 0.90)))
                if paragraph:
                    source_cap = int(
                        round(region.source_line_height * 0.92 * scale)
                    )
                    max_size = min(
                        max_size, max(self.min_font_size, source_cap)
                    )
                if region.role in {"hashtag", "metadata"}:
                    # Tags and chrome keep the source's small type, even when
                    # OCR merged several tag lines into one tall box.
                    chrome_cap = int(round(box.h * 0.80 * scale))
                    max_size = min(
                        max_size, max(self.min_font_size, chrome_cap)
                    )
                if (
                    not paragraph
                    and page_line_height > 0
                    and box.h <= page_line_height * 1.45
                ):
                    stable_cap = int(
                        round(page_line_height * 0.85 * scale)
                    )
                    max_size = min(
                        max_size, max(self.min_font_size, stable_cap)
                    )
            fit_tight = False if paragraph else None
            # Fit once with the regular face to learn the natural display size,
            # then re-fit with the weight that suits that size and scenario.
            base_fit = fit_text(
                translated,
                box,
                self.fontbook,
                target_lang=target_lang,
                source_direction=region.direction,
                min_size=self.min_font_size,
                font_scale=scale,
                max_size=max_size,
                tight=fit_tight,
            )
            weight = self._choose_weight(base_fit.font_size, mode, region)
            fit = (
                base_fit
                if weight == "regular"
                else fit_text(
                    translated,
                    box,
                    self.fontbook,
                    target_lang=target_lang,
                    source_direction=region.direction,
                    min_size=self.min_font_size,
                    font_scale=scale,
                    weight=weight,
                    max_size=max_size,
                    tight=fit_tight,
                )
            )
            foreground, stroke, stroke_width, background = analyze_text_style(
                original,
                box,
                mode=mode,
                font_size=fit.font_size,
                pixels=source_pixels,
            )
            font = self.fontbook.get(fit.font_size, weight)
            horizontal_align = self._horizontal_align(
                region, box, page_width=result.width, paragraph=paragraph, mode=mode
            )
            self._draw_fitted(
                draw,
                box,
                fit,
                font,
                foreground,
                stroke=stroke,
                stroke_width=stroke_width,
                align=horizontal_align,
                bottom_align=(
                    horizontal_align == "center"
                    and mode in {"manga", "novel", "auto"}
                    and fit.layout == "horizontal"
                    and fit.font_size >= 28
                ),
                top_align=paragraph and horizontal_align != "center",
            )
            if fit.overflow:
                warnings.append(
                    f"Text box {region.id} overflowed its original bounds even at "
                    f"{fit.font_size}px; the translation may be too long."
                )
            fit_ratio = min(
                1.0,
                (box.w - 4) / max(1, fit.width),
                (box.h - 4) / max(1, fit.height),
            )
            blocks.append(
                RenderBlock(
                    id=region.id,
                    source_text=region.text,
                    translated_text=translated,
                    bbox=[box.x, box.y, box.w, box.h],
                    polygon=region.polygon,
                    direction=region.direction,
                    role=region.role,
                    align=horizontal_align,
                    layout=fit.layout,  # type: ignore[arg-type]
                    font_size=fit.font_size,
                    line_height=fit.line_height,
                    lines=fit.lines,
                    text_color=list(foreground),
                    background_color=list(background),
                    overflow=fit.overflow,
                    fit_ratio=round(max(0.0, fit_ratio), 3),
                    font_weight=weight,
                    stroke_color=list(stroke) if stroke else None,
                    stroke_width=stroke_width,
                )
            )
        if self.fontbook.warning:
            warnings.append(self.fontbook.warning)
        return RenderOutput(image=result, blocks=blocks, warnings=warnings)

    @staticmethod
    def _choose_weight(
        font_size: int, mode: str, region: TextRegion | None = None
    ) -> str:
        """Pick a face for the rendered size and scenario.

        Large display lettering gets a poster face; ordinary manga dialogue gets
        a bold face so it reads like lettering rather than body text. Chrome and
        hashtags always keep the source's regular weight: the old size-only
        choice turned a 14 px tag row bold/display once a tall merged box
        inflated its font.
        """
        if region is not None:
            if region.role in {"hashtag", "metadata"}:
                return "regular"
            if region.source_line_height > 0 and mode in {"auto", "ui", "game"}:
                return "regular"
        if mode in {"manga", "novel"}:
            if font_size >= 30:
                return "display"
            if font_size >= 16:
                return "bold"
        elif mode == "auto":
            if font_size >= 34:
                return "display"
            if font_size >= 16:
                return "bold"
        elif mode == "game":
            if font_size >= 18:
                return "bold"
        elif mode == "ui":
            if font_size >= 24:
                return "bold"
        return "regular"

    @staticmethod
    def _horizontal_align(
        region: TextRegion,
        box: BBox,
        *,
        page_width: int,
        paragraph: bool,
        mode: str,
    ) -> str:
        """Preserve the source's left/center/right alignment.

        OCR boxes are tight, so a shorter translation drawn centered drifts
        into the middle of the original text. A row hugging the left margin is
        left-aligned, one hugging the right margin is right-aligned (timestamps,
        counters), and only genuinely inset text stays centered. Manga/novel
        lettering keeps its centered style.
        """
        if region.direction != "horizontal":
            return "center"
        if paragraph and mode in {"auto", "ui", "game"}:
            return "left"
        social = mode in {"auto", "ui", "game"} or region.role in {
            "hashtag",
            "metadata",
        }
        if social and page_width > 0:
            edge = page_width * 0.12
            left_margin = box.x
            right_margin = page_width - box.right
            if left_margin <= edge and right_margin > edge:
                return "left"
            if right_margin <= edge and left_margin > edge:
                return "right"
        return "center"

    @staticmethod
    def _draw_fitted(
        draw: ImageDraw.ImageDraw,
        box: BBox,
        fit: FitResult,
        font,
        color: tuple[int, int, int],
        *,
        stroke: tuple[int, int, int] | None = None,
        stroke_width: int = 0,
        align: str = "center",
        bottom_align: bool = False,
        top_align: bool = False,
    ) -> None:
        center_x = box.x + box.w / 2.0
        center_y = box.y + box.h / 2.0
        stroke_kwargs = (
            {"stroke_width": max(0, int(stroke_width)), "stroke_fill": stroke}
            if stroke and stroke_width > 0
            else {}
        )
        if fit.layout == "vertical":
            total_width = len(fit.lines) * fit.line_height
            start_x = center_x + total_width / 2.0 - fit.line_height / 2.0
            # Manga columns are read from the top down. Centering a short
            # translation vertically pushed it into the artwork below the
            # original lettering; anchor the first glyph at the box top.
            first_center_y = box.y + fit.line_height / 2.0 + 1
            for column_index, column in enumerate(fit.lines):
                x = start_x - column_index * fit.line_height
                for row_index, char in enumerate(column):
                    y = first_center_y + row_index * fit.line_height
                    draw.text(
                        (x, y),
                        char,
                        font=font,
                        fill=color,
                        anchor="mm",
                        **stroke_kwargs,
                    )
            return

        total_height = len(fit.lines) * fit.line_height
        if bottom_align:
            # Big titles sit on the original baseline, below caption ribbons.
            first_center = box.bottom - total_height + fit.line_height / 2.0 - 1
        elif top_align:
            # Paragraphs/tag rows grow downward from the original first line.
            first_center = box.y + fit.line_height / 2.0
        else:
            first_center = center_y - total_height / 2.0 + fit.line_height / 2.0
        if align == "left":
            text_x, text_anchor = box.x, "lm"
        elif align == "right":
            text_x, text_anchor = box.right, "rm"
        else:
            text_x, text_anchor = center_x, "mm"
        for index, line in enumerate(fit.lines):
            if not line:
                continue
            y = first_center + index * fit.line_height
            draw.text(
                (text_x, y),
                line,
                font=font,
                fill=color,
                anchor=text_anchor,
                **stroke_kwargs,
            )


def _is_cjk_char(char: str) -> bool:
    return any(
        start <= ord(char) <= end
        for start, end in ((0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xAC00, 0xD7AF))
    )
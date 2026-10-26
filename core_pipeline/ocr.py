"""OCR backends and reading-order reconstruction.

RapidOCR is the primary engine. It runs the PP-OCR detector/recognizer as
plain ONNX graphs on ONNX Runtime, so the same ``CUDAExecutionProvider`` /
``onnxruntime-gpu`` stack that accelerates LaMa also accelerates OCR. There is
no PaddlePaddle dependency and no framework import cost.

Model profiles are selected per language: RapidOCR's bundled PP-OCRv6 ``small``
models cover Chinese, English, Japanese, and the Latin languages without a
download, while PP-OCRv4/v5 mobile recognizers are used for languages the v6
multilingual graph does not cover (Korean, Cyrillic, Arabic, ...). The classic
PP-OCRv4 ONNX exports remain selectable through ``OTR_OCR_VERSION``.

Tesseract stays available as a lightweight fallback. Initialization is lazy so
the API can start in milliseconds before any model weights are loaded.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import logging
import re
import statistics
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image

from core_pipeline.errors import (
    OcrProcessingError,
    OcrUnavailableError,
    OTranslatorError,
)
from core_pipeline.runtime import activate_gpu_runtime, resolve_providers
from core_pipeline.types import OcrOutput, TextRegion

logger = logging.getLogger(__name__)

_CJK_RANGES = (
    (0x3040, 0x30FF),  # kana
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),  # hangul
)

# Canonical language codes understood by the RapidOCR model resolver. Latin
# languages share one recognizer, as do Cyrillic and Devanagari languages.
OCR_LANG_ALIASES = {
    "zh": "ch",
    "zh-cn": "ch",
    "zh-hans": "ch",
    "zh-sg": "ch",
    "cn": "ch",
    "chinese": "ch",
    "zh-tw": "chinese_cht",
    "zh-hant": "chinese_cht",
    "zh-hk": "chinese_cht",
    "chinese_cht": "chinese_cht",
    "ja": "japan",
    "jp": "japan",
    "jpn": "japan",
    "japanese": "japan",
    "japan": "japan",
    "ko": "korean",
    "kr": "korean",
    "kor": "korean",
    "korean": "korean",
    "en": "en",
    "eng": "en",
    "english": "en",
    "fr": "latin",
    "french": "latin",
    "de": "latin",
    "german": "latin",
    "es": "latin",
    "spanish": "latin",
    "it": "latin",
    "pt": "latin",
    "nl": "latin",
    "sv": "latin",
    "pl": "latin",
    "tr": "latin",
    "vi": "latin",
    "id": "latin",
    "ms": "latin",
    "ro": "latin",
    "da": "latin",
    "fi": "latin",
    "latin": "latin",
    "ru": "cyrillic",
    "russian": "cyrillic",
    "uk": "cyrillic",
    "be": "cyrillic",
    "cyrillic": "cyrillic",
    "ar": "arabic",
    "arabic": "arabic",
    "hi": "devanagari",
    "devanagari": "devanagari",
    "th": "th",
    "el": "el",
    "ta": "ta",
    "te": "te",
    "ka": "ka",
}

OCR_SUPPORTED_LANGS = frozenset(
    {
        "ch",
        "chinese_cht",
        "en",
        "japan",
        "korean",
        "latin",
        "cyrillic",
        "arabic",
        "devanagari",
        "th",
        "el",
        "ta",
        "te",
        "ka",
    }
)

# PP-OCRv6 is a single multilingual recognizer shipped inside the rapidocr
# wheel. It covers CJK + Latin; the specialized v4/v5 recognizers remain in use
# for Korean, Cyrillic, Arabic, Indic, Thai, Greek, and Georgian.
_V6_LANGS = frozenset({"ch", "chinese_cht", "en", "japan", "latin"})
_V5_LANGS = frozenset(
    {
        "ch",
        "en",
        "korean",
        "latin",
        "cyrillic",
        "arabic",
        "devanagari",
        "th",
        "el",
        "ta",
        "te",
    }
)

_OCR_VERSION_ALIASES = {
    "auto": "auto",
    "v4": "PP-OCRv4",
    "4": "PP-OCRv4",
    "ppocrv4": "PP-OCRv4",
    "pp-ocrv4": "PP-OCRv4",
    "v5": "PP-OCRv5",
    "5": "PP-OCRv5",
    "ppocrv5": "PP-OCRv5",
    "pp-ocrv5": "PP-OCRv5",
    "v6": "PP-OCRv6",
    "6": "PP-OCRv6",
    "ppocrv6": "PP-OCRv6",
    "pp-ocrv6": "PP-OCRv6",
}


def normalize_ocr_lang(value: str, default: str = "ch") -> str:
    """Map ISO-style language codes to RapidOCR's canonical language names."""
    code = (value or "").strip().lower().replace("_", "-")
    if not code or code == "auto":
        code = (default or "ch").strip().lower().replace("_", "-")
        if not code or code == "auto":
            code = "ch"
    mapped = OCR_LANG_ALIASES.get(code, code)
    if mapped in OCR_SUPPORTED_LANGS:
        return mapped
    return "ch"


# Kept for callers that imported the old name directly.
normalize_paddle_lang = normalize_ocr_lang


def _normalize_ocr_version(value: str | None) -> str:
    key = (value or "auto").strip().lower().replace(" ", "")
    return _OCR_VERSION_ALIASES.get(key, "auto")


@dataclass(frozen=True)
class RapidOCRModelSpec:
    """One detector/recognizer pair selected for a language."""

    det_version: str
    det_lang: str
    det_model_type: str
    rec_version: str
    rec_lang: str
    rec_model_type: str
    det_model_path: str | None = None
    rec_model_path: str | None = None

    @property
    def signature(self) -> tuple[str, str, str, str, str, str]:
        """Model identity; equal signatures can share one initialized backend.

        PP-OCRv6 is one multilingual recognizer for every supported language,
        so the ``rec_lang`` label is deliberately excluded from the identity.
        Otherwise ``ch`` and ``japan`` would load the same ONNX file twice.
        """
        rec_identity = "multi" if self.rec_version == "PP-OCRv6" else self.rec_lang
        return (
            self.det_version,
            self.det_lang,
            self.det_model_type,
            self.rec_version,
            rec_identity,
            self.rec_model_type,
        )

    @property
    def label(self) -> str:
        version = self.rec_version.replace("PP-OCR", "").lower()
        return f"{version}-{self.rec_model_type}"


def select_rapidocr_models(
    lang: str,
    *,
    version: str = "auto",
    profile: str = "auto",
    det_model_path: str | None = None,
    rec_model_path: str | None = None,
) -> RapidOCRModelSpec:
    """Return the RapidOCR detector/recognizer pair for a language.

    ``auto`` prefers the PP-OCRv6 multilingual graphs bundled with rapidocr
    (zero download, strong Japanese accuracy) and falls back to PP-OCRv4 mobile
    models for languages v6 does not cover. ``quality`` selects server/medium
    graphs where they exist; ``fast`` selects tiny/mobile graphs.
    """
    normalized = normalize_ocr_lang(lang)
    chosen = _normalize_ocr_version(version)
    if chosen == "auto":
        if normalized in _V6_LANGS:
            chosen = "PP-OCRv6"
        elif normalized in _V5_LANGS:
            chosen = "PP-OCRv5"
        else:
            chosen = "PP-OCRv4"
    elif (
        (chosen == "PP-OCRv6" and normalized not in _V6_LANGS)
        or (chosen == "PP-OCRv5" and normalized not in _V5_LANGS)
    ):
        chosen = "PP-OCRv4"

    profile_name = (profile or "auto").strip().lower()
    if profile_name not in {"auto", "fast", "quality"}:
        profile_name = "auto"

    if chosen == "PP-OCRv6":
        model_type = {"auto": "small", "fast": "tiny", "quality": "medium"}[
            profile_name
        ]
        det = (chosen, "ch", model_type)
        rec_lang = "en" if normalized == "latin" else normalized
        rec = (chosen, rec_lang, model_type)
    else:
        mobile = "mobile"
        server = "server"
        det_type = server if profile_name == "quality" else mobile
        rec_type = (
            server
            if profile_name == "quality" and normalized == "ch"
            else mobile
        )
        det = (chosen, "ch", det_type)
        rec = (chosen, normalized, rec_type)

    return RapidOCRModelSpec(
        det_version=det[0],
        det_lang=det[1],
        det_model_type=det[2],
        rec_version=rec[0],
        rec_lang=rec[1],
        rec_model_type=rec[2],
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
    )



def contains_kana(text: str) -> bool:
    return any(
        0x3040 <= ord(ch) <= 0x30FF or 0x31F0 <= ord(ch) <= 0x31FF
        for ch in text
    )


def contains_cjk(text: str) -> bool:
    return any(any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES) for ch in text)


_LONG_VOWEL_AFTER_KANA = re.compile(
    r"(?<=[\u3040-\u30ff])(?:1|１|一)(?=[\u3040-\u30ff])"
)
_LONG_VOWEL_AT_END = re.compile(
    r"(?<=[\u3040-\u30ff])(?:1|１|一|巻|川)(?=$|[\s、。，．！？!?…〜~])"
)


def repair_japanese_ocr(text: str) -> str:
    """Repair the most common vertical-Japanese recognition confusions.

    The PP-OCR recognizer receives rotated columns and regularly reports the
    long-vowel mark ``ー`` as ``1``/``一`` (or ``巻`` at the end of a line).
    Left alone, a line such as ``魔物狩りた1のし巻`` reaches the translator as
    "volume 1", which is exactly how hallucinated titles like
    "魔物狩猎者 1 第一卷" are produced. Only unambiguous kana contexts are
    rewritten; everything else is preserved for the translator.
    """
    if not text or not contains_kana(text):
        return text
    repaired = _LONG_VOWEL_AFTER_KANA.sub("ー", text)
    repaired = _LONG_VOWEL_AT_END.sub("ー", repaired)
    # Full-width punctuation keeps CJK measures and typesetting predictable.
    repaired = repaired.replace("?", "？").replace("!", "！")
    repaired = re.sub(r"[ \t]{2,}", " ", repaired)
    # Stray backticks/quotes are a common artifact of a long-vowel mark in
    # rotated columns ("ヒヤツハ!`").
    repaired = repaired.rstrip("`'\" ")
    return repaired.strip()


def detect_text_direction(text: str, width: float, height: float) -> str:
    """Infer a text line's layout from its box geometry.

    Vertical CJK columns are much taller than they are wide. Horizontal UI text
    is usually wider than tall; near-square boxes default to horizontal unless
    they contain only a few CJK characters.
    """
    if width <= 1 or height <= 1:
        return "horizontal"
    ratio = height / width
    if ratio >= 1.8:
        return "vertical"
    if contains_cjk(text) and ratio >= 1.25 and len(text.strip()) <= 4:
        return "vertical"
    return "horizontal"


def _polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _polygon_iou(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    from core_pipeline.types import BBox

    ba, bb = BBox.from_polygon(a), BBox.from_polygon(b)
    x0, y0 = max(ba.x, bb.x), max(ba.y, bb.y)
    x1, y1 = min(ba.right, bb.right), min(ba.bottom, bb.bottom)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = ba.area + bb.area - inter
    return inter / union if union else 0.0



def _boxes_intersection(a: TextRegion, b: TextRegion) -> tuple[int, int]:
    ba, bb = a.bbox, b.bbox
    x0, y0 = max(ba.x, bb.x), max(ba.y, bb.y)
    x1, y1 = min(ba.right, bb.right), min(ba.bottom, bb.bottom)
    return max(0, x1 - x0), max(0, y1 - y0)


def _nested_duplicate(region: TextRegion, other: TextRegion) -> bool:
    """True when ``region`` is a re-detected fragment inside ``other``.

    The detector occasionally emits the first glyph of a wide box as its own
    tiny region (``同`` beside ``同じようなポーズ…``). Its IoU is far below the
    duplicate threshold, but merging it into the wide line would duplicate the
    glyph and corrupt the translation.
    """
    text = region.text.strip()
    other_text = other.text.strip()
    if not text or not other_text or text == other_text:
        return False
    if text.casefold() not in other_text.casefold():
        return False
    area = max(1, region.bbox.area)
    other_area = max(1, other.bbox.area)
    if area > other_area * 0.55:
        return False
    _overlap_w, overlap_h = _boxes_intersection(region, other)
    if overlap_h < region.bbox.h * 0.55:
        return False
    # The duplicate often starts a few pixels before the wide box (the detector
    # re-read one glyph with its own padding). Accept a small gap as long as the
    # two boxes are visually on the same baseline.
    gap = max(region.bbox.x, other.bbox.x) - min(region.bbox.right, other.bbox.right)
    return gap <= max(2, region.bbox.w * 0.6)


def deduplicate_regions(regions: list[TextRegion]) -> list[TextRegion]:
    """Drop duplicate detections that the detector occasionally emits per line."""
    kept: list[TextRegion] = []
    for region in regions:
        duplicate = False
        for index, other in enumerate(kept):
            same_text = region.text.strip() == other.text.strip()
            if same_text and _polygon_iou(region.polygon, other.polygon) > 0.80:
                duplicate = True
                if region.confidence > other.confidence:
                    kept[index] = region
                break
            if _nested_duplicate(region, other):
                duplicate = True
                break
            if _nested_duplicate(other, region):
                # The wide box arrived after its tiny duplicate; replace it.
                kept[index] = region
                duplicate = True
                break
        if not duplicate:
            kept.append(region)
    return kept


def _group_by_axis(
    regions: list[TextRegion],
    *,
    axis: str,
    tolerance: float,
    descending: bool,
) -> list[list[TextRegion]]:
    """Group boxes into rows/columns and sort groups along the reading axis."""
    if not regions:
        return []
    groups: list[dict[str, Any]] = []
    ordered = sorted(
        regions,
        key=lambda r: (
            (r.bbox.y + r.bbox.h / 2) if axis == "y" else (r.bbox.x + r.bbox.w / 2),
            r.bbox.x,
        ),
        reverse=False,
    )
    for region in ordered:
        box = region.bbox
        center = (box.y + box.h / 2) if axis == "y" else (box.x + box.w / 2)
        target = None
        for group in groups:
            if abs(center - group["center"]) <= tolerance:
                target = group
                break
        if target is None:
            groups.append({"center": center, "items": [region]})
        else:
            target["items"].append(region)
            target["center"] = statistics.mean(
                [
                    (r.bbox.y + r.bbox.h / 2) if axis == "y" else (r.bbox.x + r.bbox.w / 2)
                    for r in target["items"]
                ]
            )
    groups.sort(key=lambda g: g["center"], reverse=descending)
    for group in groups:
        group["items"].sort(key=lambda r: r.bbox.x, reverse=descending)
    return [group["items"] for group in groups]


def merge_stacked_lines(
    regions: list[TextRegion],
    *,
    max_height: int = 24,
    max_gap_ratio: float = 0.9,
) -> list[TextRegion]:
    """Merge tiny stacked caption lines into one paragraph region.

    Light-novel captions and manga subtitle ribbons are often 6-12 px tall and
    detected as two or three separate boxes. Translating each fragment in
    isolation produces broken Chinese, and typesetting each box separately
    overlaps them. Merging the visual stack gives the translator one sentence
    and the renderer one paragraph box.
    """
    if len(regions) < 2:
        return regions
    candidates = [
        region
        for region in regions
        if region.role == "text"
        and region.direction == "horizontal"
        and region.bbox.h <= max_height
    ]
    if len(candidates) < 2:
        return regions
    clusters: list[list[TextRegion]] = []
    for region in sorted(candidates, key=lambda r: (r.bbox.y, r.bbox.x)):
        placed = False
        for cluster in clusters:
            if len(cluster) >= 4:
                continue
            if _stackable(cluster[-1], region, max_gap_ratio=max_gap_ratio):
                cluster.append(region)
                placed = True
                break
        if not placed:
            clusters.append([region])

    merged: dict[int, TextRegion] = {}
    consumed: set[int] = set()
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        cluster.sort(key=lambda r: (r.bbox.y, r.bbox.x))
        x0 = min(r.bbox.x for r in cluster)
        y0 = min(r.bbox.y for r in cluster)
        x1 = max(r.bbox.right for r in cluster)
        y1 = max(r.bbox.bottom for r in cluster)
        text = "".join(r.text for r in cluster)
        if not contains_cjk(text):
            text = " ".join(r.text for r in cluster)
        merged[cluster[0].id] = TextRegion(
            id=cluster[0].id,
            text=text.strip(),
            confidence=round(min(r.confidence for r in cluster), 4),
            polygon=[
                (float(x0), float(y0)),
                (float(x1), float(y0)),
                (float(x1), float(y1)),
                (float(x0), float(y1)),
            ],
            direction="horizontal",
            mask_expand_px=max(8, int(round(max(r.bbox.h for r in cluster) * 0.8))),
            source_line_height=round(
                float(statistics.median([r.bbox.h for r in cluster])), 2
            ),
        )
        consumed.update(r.id for r in cluster[1:])

    if not merged:
        return regions
    result: list[TextRegion] = []
    for region in regions:
        if region.id in merged:
            result.append(merged[region.id])
        elif region.id not in consumed:
            result.append(region)
    for index, region in enumerate(result):
        region.id = index
    return result


def _stackable(
    upper: TextRegion, lower: TextRegion, *, max_gap_ratio: float = 0.9
) -> bool:
    if upper.direction != "horizontal" or lower.direction != "horizontal":
        return False
    a, b = upper.bbox, lower.bbox
    smaller_h = max(1, min(a.h, b.h))
    larger_h = max(1, max(a.h, b.h))
    if larger_h / smaller_h > 1.9:
        return False
    overlap = min(a.right, b.right) - max(a.x, b.x)
    if overlap < min(a.w, b.w) * 0.55:
        return False
    # Stacked lines of one paragraph share a margin (or a centre line). Two
    # fragments of different columns can overlap by >55% while their left edges
    # are far apart; stacking them welded two source lines into one box whose
    # translation then overprinted the neighbouring block.
    smaller_w = max(1, min(a.w, b.w))
    larger_w = max(1, max(a.w, b.w))
    left_delta = abs(a.x - b.x)
    right_delta = abs(a.right - b.right)
    center_delta = abs((a.x + a.w / 2.0) - (b.x + b.w / 2.0))
    aligned = (
        left_delta <= max(4, smaller_w * 0.2)
        or right_delta <= max(4, smaller_w * 0.2)
        or center_delta <= max(4, larger_w * 0.15)
    )
    if not aligned:
        return False
    gap = b.y - a.bottom
    if gap < -smaller_h * 0.5:
        return False
    return not gap > max(3.0, smaller_h * max_gap_ratio)


def _meaningful_text(text: str) -> bool:
    """Reject punctuation/line fragments produced by over-eager detection."""
    return any(char.isalnum() or contains_cjk(char) for char in text)


# A social/UI line that only holds a relative timestamp ("4小时", "4h",
# "2d ago", "昨日") is chrome, not post copy. The regex also accepts absolute
# clock labels because a status-bar "23:48" must never join a caption.
_RELATIVE_TIME_RE = re.compile(
    r"\b(?:just\s+now|yesterday|today)\b"
    r"|\b\d{1,2}:\d{2}(?::\d{2})?\b"
    r"|\b\d+\s*(?:s|sec|secs|m|min|mins|h|hr|hrs|d|day|days|w|wk|wks|y|yr|yrs)"
    r"(?![A-Za-z])"
    r"|\d+\s*(?:秒|秒钟|分钟|分|小时|時間|时间|天|日|周|週|星期|个月|月|年)(?:前)?"
    r"|\d+\s*(?:초|분|시간|일|주|개월|년)\s*(?:전)?",
    re.IGNORECASE,
)

# Tag-only lines (``#cosmichorror #cthulhu``) need their own small block; the
# leading # makes them trivial to separate from prose that merely mentions one.
_HASHTAG_TOKEN_RE = re.compile(r"#[^\s#]+", re.UNICODE)

# Counters are language-neutral chrome: ``18``, ``390``, ``8.2万``, ``1.2K
# views``. Translating or redrawing them buys nothing and the OCR box usually
# also contains the neighbour icon (repost/like), so the mask must skip them
# entirely. Digits inside prose never reach this check because the region text
# contains the sentence around them.
_COUNTER_NOISE_RE = re.compile(
    r"[\d\s.,'’·•_+\-–—/\\:：;；()（）\[\]【】%％＋★☆♥❤]"
)
_COUNTER_UNIT_RE = re.compile(
    r"(?:[kKmMbBwW]|万|萬|亿|億|千|百"
    r"|次|条|件|人|个|份|张|位"
    r"|views?|likes?|reposts?|retweets?|replies?|comments?"
    r"|followers?|following|plays?|shares?|bookmarks?)",
    re.IGNORECASE,
)
# PP-OCR often reads the action icon in front of a count as a one- or two-letter
# token: "Q 81" (comment bubble), "ll 8.2万" (bar-chart). Strip that prefix
# before deciding, otherwise the counter is translated and its mask erases the
# very icon that produced the stray letters.
_COUNTER_ICON_PREFIX_RE = re.compile(r"^[A-Za-z]{1,2}\s*(?=[+\-–—]?\s*\d)")

# A social header is a display name, an @handle, and a relative/absolute
# timestamp on one line: ``Mario Zechner@badlogicgames·10月17日``. Generic
# timestamp detection rejects it because of the length limit; treating it as
# metadata keeps the verified badge and avatar pixels intact.
_SOCIAL_HEADER_RE = re.compile(
    r"^\s*[^\n@]{1,32}@[A-Za-z0-9._]{2,32}"
    r"(?:\s*[·•|丨]\s*|\s+)"
    r"(?:"
    r"\d{1,2}\s*(?:秒|秒钟|分钟|分|小时|時間|时间|天|日|周|週|星期|个月|月|年)(?:前)?"
    r"|\d{1,2}\s*月\s*\d{1,2}\s*日"
    r"|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d+\s*(?:s|m|h|d|w|y)(?![A-Za-z])"
    r"|\d{1,2}:\d{2}(?::\d{2})?"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}"
    r"(?:,\s*\d{4})?"
    r"|just\s+now|yesterday|today"
    r")\s*$",
    re.IGNORECASE,
)


def _hashtag_only(text: str) -> bool:
    value = (text or "").strip()
    tokens = _HASHTAG_TOKEN_RE.findall(value)
    if not tokens:
        return False
    remainder = _HASHTAG_TOKEN_RE.sub("", value)
    remainder = re.sub(r"[\s\W_]+", "", remainder, flags=re.UNICODE)
    # A tag row may contain separators or a short label ("Tags:"), but not a
    # sentence's worth of words around the tags.
    return len(remainder) <= max(2, len(tokens))


def _looks_like_handle(text: str) -> bool:
    """True for an Instagram/Twitter-style @handle or underscore username."""
    value = (text or "").strip()
    if not (4 <= len(value) <= 32) or any(char.isspace() for char in value):
        return False
    if value.startswith("@"):
        return bool(re.fullmatch(r"@[A-Za-z0-9._]+", value))
    # OCR sometimes glues a misrecognized timestamp suffix onto the handle
    # (a Cyrillic "4h" instead of the CJK one). Recognize the leading handle
    # token even when the tail is not Latin.
    prefix = re.match(r"^(@?[A-Za-z0-9._]{4,32})", value)
    if prefix and prefix.end() < len(value):
        token = prefix.group(1)
        if len(token) >= 8 and (token.startswith("@") or "_" in token):
            return True
    if not re.fullmatch(r"[A-Za-z0-9._]+", value):
        return False
    if "_" in value:
        # "ai_cosmic_horror_art" is a username; "Chapter_4" is more likely a
        # title fragment, so require at least a short, mostly lowercase token.
        return len(value) >= 8 and (
            not any(char.isupper() for char in value) or len(value) >= 12
        )
    return bool(
        "." in value
        and len(value) >= 8
        and not any(char.isupper() for char in value)
    )


def _looks_like_timestamp(text: str) -> bool:
    value = (text or "").strip()
    if not value or len(value) > 32:
        return False
    return bool(_RELATIVE_TIME_RE.search(value))


def _looks_like_social_header(text: str) -> bool:
    """True for ``Display Name@handle·10月17日`` style rows."""
    value = (text or "").strip()
    if not value or len(value) > 72:
        return False
    return bool(_SOCIAL_HEADER_RE.match(value))


def _looks_like_counter(text: str) -> bool:
    """True when a region is only a number/count, not prose.

    ``18``, ``390``, ``8.2万``, ``12,345``, ``1.2K views`` and ``99+`` are
    language-neutral. They must not be translated, masked, or redrawn: their
    OCR box often also contains the neighbouring UI icon (repost, like), and a
    rectangular mask would erase that icon. Digits embedded in a sentence are
    unaffected because the surrounding words survive the filter.
    """
    value = (text or "").strip()
    if not value or len(value) > 32:
        return False
    if not any(char.isdigit() for char in value):
        return False
    value = _COUNTER_ICON_PREFIX_RE.sub("", value)
    residue = _COUNTER_NOISE_RE.sub("", value)
    residue = _COUNTER_UNIT_RE.sub("", residue)
    return not residue.strip()


def classify_region_roles(regions: list[TextRegion]) -> list[TextRegion]:
    """Annotate social/UI chrome before paragraph reconstruction.

    Screenshot OCR has no semantic layer: a username row, its relative
    timestamp, the caption, and the hashtags all look like horizontal lines with
    similar heights. Geometry-only paragraph merging therefore welds
    ``ai_cosmic_horror_art 4小时`` onto the first caption line. Classifying the
    chrome first keeps those rows out of the translation paragraph and lets the
    renderer keep tags at their original small scale.
    """
    if not regions:
        return regions
    for region in regions:
        if _hashtag_only(region.text):
            region.role = "hashtag"

    horizontal = [
        region
        for region in regions
        if region.direction == "horizontal" and region.text.strip()
    ]
    if horizontal:
        heights = [max(1, region.bbox.h) for region in horizontal]
        tolerance = max(4.0, statistics.median(heights) * 0.35)
        for row in _group_by_axis(
            horizontal, axis="y", tolerance=tolerance, descending=False
        ):
            if all(region.role == "hashtag" for region in row):
                continue
            content = [region for region in row if region.role != "hashtag"]
            if not content:
                continue
            combined = " ".join(region.text.strip() for region in content)
            is_metadata = (
                _looks_like_social_header(combined)
                or any(
                    _looks_like_social_header(region.text) for region in content
                )
                or _looks_like_timestamp(combined)
                or any(
                    _looks_like_timestamp(region.text)
                    or _looks_like_handle(region.text)
                    for region in content
                )
            )
            if is_metadata:
                for region in content:
                    region.role = "metadata"

    # Counters are classified last: a row that also carries a handle or a
    # timestamp is chrome already, and ``keep`` must not override that.
    for region in regions:
        if region.role == "text" and _looks_like_counter(region.text):
            region.role = "keep"
    return regions


def merge_title_pieces(regions: list[TextRegion]) -> list[TextRegion]:
    """Join pieces of one horizontal title that the detector split apart.

    A title such as ``偽聖女でした`` is often detected as a large ``偽聖女`` box
    plus a smaller ``でした`` box on the same baseline. Translating and setting
    them separately detaches the sentence ending (the old output showed a lone
    "!" at the far right). Merge only same-baseline neighbours, so the small
    caption ribbon above is not swallowed.
    """
    if len(regions) < 2:
        return regions
    candidates = [
        r for r in regions if r.role == "text" and r.direction == "horizontal"
    ]
    parent = {id(region): id(region) for region in candidates}

    def find(key: int) -> int:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            if _title_piece_pair(first, second):
                union(id(first), id(second))

    clusters: dict[int, list[TextRegion]] = {}
    for region in candidates:
        clusters.setdefault(find(id(region)), []).append(region)
    merged: dict[int, TextRegion] = {}
    consumed: set[int] = set()
    for cluster in clusters.values():
        if len(cluster) < 2:
            continue
        cluster.sort(key=lambda r: r.bbox.x)
        x0 = min(r.bbox.x for r in cluster)
        y0 = min(r.bbox.y for r in cluster)
        x1 = max(r.bbox.right for r in cluster)
        y1 = max(r.bbox.bottom for r in cluster)
        # Keep the union box: the redrawn title can use the full original
        # baseline width, and the renderer bottom-aligns large horizontal
        # titles so they stay below a caption ribbon.
        merged[cluster[0].id] = TextRegion(
            id=cluster[0].id,
            text="".join(r.text for r in cluster).strip(),
            confidence=round(min(r.confidence for r in cluster), 4),
            polygon=[
                (float(x0), float(y0)),
                (float(x1), float(y0)),
                (float(x1), float(y1)),
                (float(x0), float(y1)),
            ],
            direction="horizontal",
        )
        consumed.update(r.id for r in cluster[1:])
    if not merged:
        return regions
    result: list[TextRegion] = []
    for region in regions:
        if region.id in merged:
            result.append(merged[region.id])
        elif region.id not in consumed:
            result.append(region)
    for index, region in enumerate(result):
        region.id = index
    return result


def _title_piece_pair(first: TextRegion, second: TextRegion) -> bool:
    a, b = first.bbox, second.bbox
    if max(a.h, b.h) < 20:
        return False
    first_text = first.text.strip()
    second_text = second.text.strip()
    if len(first_text) > 8 or len(second_text) > 8:
        # Long lines are sentences, not title pieces. Joining ``同`` to
        # ``同じようなポーズをしていて`` duplicated the glyph and corrupted
        # both the translation and the box geometry.
        return False
    if min(len(first_text), len(second_text)) > 5:
        return False
    if first_text and second_text and (
        first_text in second_text or second_text in first_text
    ):
        # A detector re-reading of one glyph is a duplicate, not a title piece.
        return False
    # Same baseline: bottom edges must line up. A caption ribbon sitting above
    # the title fails this test and stays separate.
    bottom_delta = abs(a.bottom - b.bottom)
    if bottom_delta > max(5, int(max(a.h, b.h) * 0.28)):
        return False
    vertical_overlap = min(a.bottom, b.bottom) - max(a.y, b.y)
    if vertical_overlap < min(a.h, b.h) * 0.55:
        return False
    gap = max(a.x, b.x) - min(a.right, b.right)
    return not gap > max(8, int(max(a.h, b.h) * 0.35))


# A line that starts with a bullet/number is a new list item, not a wrapped
# continuation of the line above it.
_PARAGRAPH_BULLET_RE = re.compile(
    r"^\s*(?:[-–—*•·●○◦▪▫‣⁃]|\d{1,2}[.)、]|[（(]\d{1,2}[)）])\s*"
)

# Joining OCR lines: CJK text wraps without spaces, while Latin text needs one.
# Sentence punctuation is handled explicitly because the last character of the
# previous line may be a quote or a dash rather than a letter.
_JOIN_NO_SPACE_AFTER = frozenset("([{“‘«〈《【「『")
_JOIN_NO_SPACE_BEFORE = frozenset(".,;:!?)]}%”’»〉》】」』、。，．！？：；…")


def _starts_paragraph_bullet(text: str) -> bool:
    return bool(_PARAGRAPH_BULLET_RE.match(text))


def _join_paragraph_text(regions: list[TextRegion]) -> str:
    """Join wrapped OCR lines into one paragraph without inventing spaces."""
    result = ""
    for region in regions:
        piece = region.text.strip()
        if not piece:
            continue
        if not result:
            result = piece
            continue
        if result.endswith(("-", "‐", "‑")) and piece[0].isalnum():
            # A word hyphenated across two physical lines.
            result = result[:-1] + piece
            continue
        left, right = result[-1], piece[0]
        if (
            contains_cjk(left)
            or contains_cjk(right)
            or left in _JOIN_NO_SPACE_AFTER
            or right in _JOIN_NO_SPACE_BEFORE
        ):
            result += piece
        else:
            result += " " + piece
    return result.strip()


def _same_line_fragment(first: TextRegion, second: TextRegion) -> bool:
    """Whether two boxes are overlapping fragments of one visual line.

    Screenshot OCR (notably the CUDA detector on small, low-contrast captures)
    sometimes returns several boxes for one physical line. They sit on the same
    baseline and either overlap or nearly touch. Typesetting them separately
    paints two translations on top of each other, so they must become one line
    *before* the paragraph passes run.
    """
    if first.direction != "horizontal" or second.direction != "horizontal":
        return False
    if first.role != second.role:
        return False
    if not first.text.strip() or not second.text.strip():
        return False
    a, b = first.bbox, second.bbox
    if a.h <= 0 or b.h <= 0 or a.w <= 0 or b.w <= 0:
        return False
    smaller_h = float(min(a.h, b.h))
    larger_h = float(max(a.h, b.h))
    if larger_h / smaller_h > 2.2:
        return False
    overlap_y = min(a.bottom, b.bottom) - max(a.y, b.y)
    if overlap_y < smaller_h * 0.55:
        return False
    center_delta = abs((a.y + a.h / 2.0) - (b.y + b.h / 2.0))
    if center_delta > max(3.0, larger_h * 0.45):
        return False
    # Only merge boxes that overlap or are separated by less than a glyph;
    # distinct labels on one toolbar ("AETHERIA" + "CHAPTER IV") stay apart.
    gap = max(a.x, b.x) - min(a.right, b.right)
    if gap > max(4.0, smaller_h * 0.6):
        return False
    overlap_x = min(a.right, b.right) - max(a.x, b.x)
    # A box almost entirely inside the other is a nested duplicate, not a
    # fragment continuation; deduplication already removed it.
    return not (overlap_x > min(a.w, b.w) * 0.85)


def merge_line_fragments(regions: list[TextRegion]) -> list[TextRegion]:
    """Join same-baseline fragments that overlap or almost touch.

    This is the layout repair for fragmented page detection. It is deliberately
    conservative: boxes must share a baseline and be adjacent (or overlapping),
    so separate UI labels are left alone. Without it, two fragments of the same
    source line are translated into two blocks whose union boxes overlap and
    whose glyphs are painted on top of each other.
    """
    if len(regions) < 2:
        return regions
    candidates = [
        region
        for region in regions
        if region.direction == "horizontal"
        and region.role in {"text", "hashtag"}
        and region.text.strip()
    ]
    if len(candidates) < 2:
        return regions

    parent = {id(region): id(region) for region in candidates}

    def find(key: int) -> int:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            if _same_line_fragment(first, second):
                union(id(first), id(second))

    clusters: dict[int, list[TextRegion]] = {}
    for region in candidates:
        clusters.setdefault(find(id(region)), []).append(region)
    merged: dict[int, TextRegion] = {}
    consumed: set[int] = set()
    for cluster in clusters.values():
        if len(cluster) < 2:
            continue
        cluster.sort(key=lambda r: (r.bbox.x, r.bbox.y))
        x0 = min(r.bbox.x for r in cluster)
        y0 = min(r.bbox.y for r in cluster)
        x1 = max(r.bbox.right for r in cluster)
        y1 = max(r.bbox.bottom for r in cluster)
        line_heights = [r.source_line_height for r in cluster if r.source_line_height > 0]
        if not line_heights:
            # Same-baseline fragments: their box heights are one source line.
            line_heights = [float(r.bbox.h) for r in cluster if r.bbox.h > 0]
        merged[cluster[0].id] = TextRegion(
            id=cluster[0].id,
            text=_join_paragraph_text(cluster),
            confidence=round(min(r.confidence for r in cluster), 4),
            polygon=[
                (float(x0), float(y0)),
                (float(x1), float(y0)),
                (float(x1), float(y1)),
                (float(x0), float(y1)),
            ],
            direction="horizontal",
            role=cluster[0].role,
            source_line_height=(
                round(statistics.median(line_heights), 2) if line_heights else 0.0
            ),
        )
        consumed.update(r.id for r in cluster[1:])

    if not merged:
        return regions
    result: list[TextRegion] = []
    for region in regions:
        if region.id in merged:
            result.append(merged[region.id])
        elif region.id not in consumed:
            result.append(region)
    for index, region in enumerate(result):
        region.id = index
    return result


def resolve_overlapping_regions(regions: list[TextRegion]) -> list[TextRegion]:
    """Merge text regions whose source boxes still overlap substantially.

    A last-resort layout guard: whatever produced the OCR boxes (GPU detector
    quirks, a merged caption ribbon, a rotated column), two translated blocks
    covering the same pixels will always overprint each other. Merge the union
    into one block so the renderer sets a single translation; roles other than
    ``text`` are untouched because chrome is never redrawn.
    """
    candidates = [
        region
        for region in regions
        if region.role == "text"
        and region.direction == "horizontal"
        and region.text.strip()
    ]
    if len(candidates) < 2:
        return regions

    parent = {id(region): id(region) for region in candidates}

    def find(key: int) -> int:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            overlap_w, overlap_h = _boxes_intersection(first, second)
            if overlap_w <= 0 or overlap_h <= 0:
                continue
            intersection = overlap_w * overlap_h
            smaller = min(first.bbox.area, second.bbox.area)
            if smaller <= 0 or intersection / smaller < 0.22:
                continue
            # Avoid chaining a sprawling title and a caption that happen to
            # overlap: only boxes sitting on the same visual band belong to one
            # overprinted block.
            center_delta = abs(
                (first.bbox.y + first.bbox.h / 2.0)
                - (second.bbox.y + second.bbox.h / 2.0)
            )
            if center_delta > max(8.0, 1.2 * max(first.bbox.h, second.bbox.h)):
                continue
            union(id(first), id(second))

    clusters: dict[int, list[TextRegion]] = {}
    for region in candidates:
        clusters.setdefault(find(id(region)), []).append(region)
    merged: dict[int, TextRegion] = {}
    consumed: set[int] = set()
    for cluster in clusters.values():
        if len(cluster) < 2:
            continue
        cluster.sort(key=lambda r: (r.bbox.y, r.bbox.x))
        x0 = min(r.bbox.x for r in cluster)
        y0 = min(r.bbox.y for r in cluster)
        x1 = max(r.bbox.right for r in cluster)
        y1 = max(r.bbox.bottom for r in cluster)
        line_heights = [r.source_line_height for r in cluster if r.source_line_height > 0]
        merged[cluster[0].id] = TextRegion(
            id=cluster[0].id,
            text=_join_paragraph_text(cluster),
            confidence=round(min(r.confidence for r in cluster), 4),
            polygon=[
                (float(x0), float(y0)),
                (float(x1), float(y0)),
                (float(x1), float(y1)),
                (float(x0), float(y1)),
            ],
            direction="horizontal",
            role="text",
            mask_expand_px=max((r.mask_expand_px for r in cluster), default=0),
            source_line_height=(
                round(statistics.median(line_heights), 2) if line_heights else 0.0
            ),
        )
        consumed.update(r.id for r in cluster[1:])

    if not merged:
        return regions
    result: list[TextRegion] = []
    for region in regions:
        if region.id in merged:
            result.append(merged[region.id])
        elif region.id not in consumed:
            result.append(region)
    for index, region in enumerate(result):
        region.id = index
    return result


def _paragraph_pair(upper: TextRegion, lower: TextRegion) -> bool:
    """Whether ``lower`` is the wrapped continuation of ``upper``."""
    if upper.direction != "horizontal" or lower.direction != "horizontal":
        return False
    # A caption never continues a username row, and a hashtag block never
    # continues the caption (the tag row keeps its own smaller type size).
    if upper.role != lower.role or upper.role not in {"text", "hashtag"}:
        return False
    if not upper.text.strip() or not lower.text.strip():
        return False
    a, b = upper.bbox, lower.bbox
    smaller = max(1, min(a.h, b.h))
    if smaller < 8:
        return False
    # A heading or a different text style has a visibly different line height.
    if max(a.h, b.h) / smaller > 1.4:
        return False
    gap = b.y - a.bottom
    if gap < -smaller * 0.4:
        return False
    if gap > max(4.0, smaller * 0.7):
        return False
    # A wrapped line never gets much wider than the line above it. A header or
    # username followed by the body does exactly that, so keep them separate.
    if b.w > a.w * 1.35:
        return False
    # Same guard for a short label (e.g. a username) followed by a wide body.
    if a.w < max(a.w, b.w) * 0.45 and len(upper.text.strip()) < 16:
        return False
    left_delta = abs(a.x - b.x)
    right_delta = abs(a.right - b.right)
    center_delta = abs((a.x + a.w / 2.0) - (b.x + b.w / 2.0))
    aligned = (
        left_delta <= max(3.0, smaller * 0.6)
        or right_delta <= max(3.0, smaller * 0.6)
        or center_delta <= max(4.0, max(a.w, b.w) * 0.22)
    )
    if not aligned:
        return False
    # The first line of a new list item must not be swallowed by the previous
    # item's paragraph.
    return not _starts_paragraph_bullet(lower.text)


def _paragraph_pair_fits(cluster: list[TextRegion], region: TextRegion) -> bool:
    if not cluster or len(cluster) >= 40:
        return False
    if not _paragraph_pair(cluster[-1], region):
        return False
    widest = max([item.bbox.w for item in cluster] + [region.bbox.w])
    if widest <= 0:
        return False
    # Inner lines of one paragraph are all roughly as wide as its widest line.
    # This rejects "short header + long body" clusters that happen to share a
    # left margin and line height.
    for existing in cluster:
        if (
            existing.bbox.w < widest * 0.45
            and len(existing.text.strip()) < 16
        ):
            return False
    # A short line is normally the last line of a paragraph; do not continue
    # past it into what is probably the next paragraph.
    return not (len(cluster) >= 2 and cluster[-1].bbox.w < widest * 0.45)


def _paragraph_cluster_ok(
    cluster: list[TextRegion], page_width: int | None
) -> bool:
    if len(cluster) < 2:
        return False
    texts = [region.text.strip() for region in cluster]
    if sum(len(text) for text in texts) < 12:
        return False
    if max(len(text) for text in texts) < 6:
        return False
    widest = max(region.bbox.w for region in cluster)
    # A few short UI labels stacked in a narrow sidebar are not a paragraph.
    return not (page_width and widest < page_width * 0.20)


def merge_paragraph_lines(
    regions: list[TextRegion],
    *,
    page_width: int | None = None,
) -> list[TextRegion]:
    """Reconstruct multi-line paragraphs from per-line OCR boxes.

    Screenshot OCR returns one box per physical line. A social-media post, a
    novel caption, or a wrapped chat message therefore arrives as several
    fragments; sending those fragments to the translation API produces
    line-by-line translations that start and stop mid-sentence. This groups
    tightly stacked, consistently aligned lines into one paragraph region so
    the model receives a complete sentence and the renderer sets one block.

    Only horizontal text is merged. Vertical manga columns are already one
    region per column and are handled by the existing caption/title mergers.
    """
    if len(regions) < 2:
        return regions
    candidates = [
        region
        for region in regions
        if region.role in {"text", "hashtag"}
        and region.direction == "horizontal"
        and region.text.strip()
        and region.bbox.h >= 8
    ]
    if len(candidates) < 2:
        return regions

    clusters: list[list[TextRegion]] = []
    for region in sorted(candidates, key=lambda r: (r.bbox.y, r.bbox.x)):
        for cluster in clusters:
            if _paragraph_pair_fits(cluster, region):
                cluster.append(region)
                break
        else:
            clusters.append([region])

    merged: dict[int, TextRegion] = {}
    consumed: set[int] = set()
    for cluster in clusters:
        if not _paragraph_cluster_ok(cluster, page_width):
            continue
        cluster.sort(key=lambda r: (r.bbox.y, r.bbox.x))
        x0 = min(r.bbox.x for r in cluster)
        y0 = min(r.bbox.y for r in cluster)
        x1 = max(r.bbox.right for r in cluster)
        y1 = max(r.bbox.bottom for r in cluster)
        merged[cluster[0].id] = TextRegion(
            id=cluster[0].id,
            text=_join_paragraph_text(cluster),
            confidence=round(min(r.confidence for r in cluster), 4),
            polygon=[
                (float(x0), float(y0)),
                (float(x1), float(y0)),
                (float(x1), float(y1)),
                (float(x0), float(y1)),
            ],
            direction="horizontal",
            role=cluster[0].role,
            source_line_height=round(
                float(statistics.median([r.bbox.h for r in cluster])), 2
            ),
        )
        consumed.update(r.id for r in cluster[1:])

    if not merged:
        return regions
    result: list[TextRegion] = []
    for region in regions:
        if region.id in merged:
            result.append(merged[region.id])
        elif region.id not in consumed:
            result.append(region)
    for index, region in enumerate(result):
        region.id = index
    return result


def order_regions(
    regions: list[TextRegion],
    *,
    mode: str = "auto",
    reading_order: str = "auto",
    source_lang: str = "auto",
) -> list[TextRegion]:
    """Assign ids in human reading order.

    Manga columns are read right-to-left. Game UIs and novels default to
    top-to-bottom, left-to-right. A vertical majority in auto mode switches to
    right-to-left columns.
    """
    if not regions:
        return []
    vertical = sum(1 for r in regions if r.direction == "vertical")
    vertical_ratio = vertical / len(regions)
    rtl = reading_order == "rtl"
    if reading_order == "auto":
        rtl = mode == "manga" or (
            mode == "auto" and vertical_ratio >= 0.35
        ) or (mode == "auto" and source_lang in {"ja"} and vertical_ratio >= 0.18)

    ordered: list[TextRegion] = []
    if vertical_ratio >= 0.5:
        # Mixed pages: manga columns are read right-to-left, while captions and
        # titles are horizontal and read top-to-bottom, left-to-right. Grouping
        # every box by x (the old behavior) put a horizontal title's boxes in
        # RTL column order and split its sentence.
        vertical_regions = [r for r in regions if r.direction == "vertical"]
        horizontal_regions = [r for r in regions if r.direction != "vertical"]
        widths = [max(1, r.bbox.w) for r in vertical_regions] or [20]
        tolerance = max(4.0, statistics.median(widths) * 0.45)
        columns = _group_by_axis(
            vertical_regions, axis="x", tolerance=tolerance, descending=rtl
        )
        vertical_ordered: list[TextRegion] = []
        for column in columns:
            column.sort(key=lambda r: r.bbox.y)
            vertical_ordered.extend(column)
        horizontal_ordered = _order_horizontal_rows(horizontal_regions)
        ordered = _merge_by_top(vertical_ordered, horizontal_ordered)
    else:
        heights = [max(1, r.bbox.h) for r in regions]
        tolerance = max(6.0, statistics.median(heights) * 0.55)
        rows = _group_by_axis(regions, axis="y", tolerance=tolerance, descending=False)
        for row in rows:
            row.sort(key=lambda r: r.bbox.x, reverse=rtl)
            ordered.extend(row)

    for index, region in enumerate(ordered):
        region.id = index
    return ordered


def _order_horizontal_rows(regions: list[TextRegion]) -> list[TextRegion]:
    if not regions:
        return []
    heights = [max(1, r.bbox.h) for r in regions]
    tolerance = max(3.0, statistics.median(heights) * 0.4)
    rows = _group_by_axis(regions, axis="y", tolerance=tolerance, descending=False)
    ordered: list[TextRegion] = []
    for row in rows:
        row.sort(key=lambda r: r.bbox.x)
        ordered.extend(row)
    return ordered


def _merge_by_top(
    vertical: list[TextRegion], horizontal: list[TextRegion]
) -> list[TextRegion]:
    """Interleave RTL columns and LTR captions by their top edge."""
    merged: list[TextRegion] = []
    i = j = 0
    while i < len(vertical) and j < len(horizontal):
        if vertical[i].bbox.y <= horizontal[j].bbox.y:
            merged.append(vertical[i])
            i += 1
        else:
            merged.append(horizontal[j])
            j += 1
    merged.extend(vertical[i:])
    merged.extend(horizontal[j:])
    return merged



class OCRBackend(Protocol):
    name: str

    def read(self, image: Image.Image) -> OcrOutput: ...


def _fragmentation_score(regions: list[TextRegion]) -> tuple[int, int]:
    """Count same-baseline fragment pairs and how many of them overlap.

    ``(pairs, overlapping)``. A clean page has one box per visual line, so
    ``pairs`` stays at zero (or one for a legitimate one-row label group).
    Fragmented detection returns several near-touching boxes per line.
    """
    pairs = 0
    overlapping = 0
    for index, first in enumerate(regions):
        for second in regions[index + 1 :]:
            if first.direction != "horizontal" or second.direction != "horizontal":
                continue
            if not first.text.strip() or not second.text.strip():
                continue
            a, b = first.bbox, second.bbox
            if a.h <= 0 or b.h <= 0 or a.w <= 0 or b.w <= 0:
                continue
            smaller_h = float(min(a.h, b.h))
            if max(a.h, b.h) / smaller_h > 2.2:
                continue
            overlap_y = min(a.bottom, b.bottom) - max(a.y, b.y)
            if overlap_y < smaller_h * 0.55:
                continue
            center_delta = abs((a.y + a.h / 2.0) - (b.y + b.h / 2.0))
            if center_delta > max(3.0, max(a.h, b.h) * 0.45):
                continue
            gap = max(a.x, b.x) - min(a.right, b.right)
            if gap > max(6.0, smaller_h * 0.6):
                continue
            overlap_x = min(a.right, b.right) - max(a.x, b.x)
            if overlap_x > min(a.w, b.w) * 0.85:
                # Nested duplicate, not a line fragment.
                continue
            pairs += 1
            if gap < 0:
                overlapping += 1
    return pairs, overlapping


def _looks_fragmented(regions: list[TextRegion]) -> bool:
    pairs, overlapping = _fragmentation_score(regions)
    return pairs >= 2 and overlapping >= 1


class RapidOCRBackend:
    """RapidOCR (PP-OCR ONNX) backend with language-aware model selection.

    Detection runs on the native page (the detector downscales to
    ``det_limit_side_len`` internally) and every detected line is cropped from
    the native image. Small crops are upscaled before recognition, tall vertical
    columns are rotated CCW, and all crops of one page are recognized in a
    single batched call. No PaddlePaddle import or graph conversion is involved:
    the PP-OCR models are plain ONNX files executed by ONNX Runtime, so the
    shared CUDA/DirectML provider policy applies unchanged.
    """

    name = "rapidocr"

    def __init__(
        self,
        *,
        lang: str = "ch",
        version: str = "auto",
        profile: str = "auto",
        models_dir: str | Path | None = None,
        det_model_path: str | None = None,
        rec_model_path: str | None = None,
        det_limit_side_len: int = 960,
        min_confidence: float = 0.5,
        device: str = "auto",
        providers: str | None = None,
        rotate_vertical: bool = True,
        cpu_fallback: bool = True,
    ) -> None:
        self.lang = normalize_ocr_lang(lang)
        self.version = _normalize_ocr_version(version)
        self.profile = str(profile or "auto").strip().lower()
        if self.profile not in {"auto", "fast", "quality"}:
            self.profile = "auto"
        self.models_dir = Path(models_dir) if models_dir else None
        self.spec = select_rapidocr_models(
            self.lang,
            version=self.version,
            profile=self.profile,
            det_model_path=det_model_path,
            rec_model_path=rec_model_path,
        )
        self.det_limit_side_len = max(320, int(det_limit_side_len))
        self.min_confidence = min_confidence
        self.rotate_vertical = rotate_vertical
        self.providers_requested = providers
        self.providers = resolve_providers(device, explicit=providers)
        self.use_cuda = "CUDAExecutionProvider" in self.providers
        self.use_dml = "DmlExecutionProvider" in self.providers
        if "OpenVINOExecutionProvider" in self.providers and not (
            self.use_cuda or self.use_dml
        ):
            logger.warning(
                "RapidOCR's onnxruntime backend does not expose "
                "OpenVINOExecutionProvider; OCR will use CPUExecutionProvider. "
                "Use OTR_OCR_DEVICE=cuda/dml or install onnxruntime-gpu/"
                "onnxruntime-directml for OCR acceleration."
            )
        self.active_providers: list[str] = []
        self.cpu_fallback = bool(cpu_fallback)
        self._engine: Any | None = None
        self._detector: Any | None = None
        self._recognizer: Any | None = None
        self._cpu_engine: Any | None = None
        self._cpu_detector: Any | None = None
        self._lock = threading.RLock()

    @staticmethod
    def is_available() -> bool:
        return bool(importlib.util.find_spec("rapidocr")) and bool(
            importlib.util.find_spec("onnxruntime")
        )

    def model_signature(self) -> tuple[str, str, str, str, str, str]:
        """Model identity this backend loads; equal signatures can be shared."""
        return self.spec.signature

    def can_refine_as_japanese(self) -> bool:
        """True when the Japanese pass would only add the zoomed refine step.

        The PP-OCRv6 multilingual graph is shared by ``ch``/``japan``/
        ``chinese_cht``/Latin, so the second OCR pass is redundant and the
        already-recognized page is refined in place instead.
        """
        return self.spec.signature == select_rapidocr_models(
            "japan",
            version=self.version,
            profile=self.profile,
            det_model_path=self.spec.det_model_path,
            rec_model_path=self.spec.rec_model_path,
        ).signature

    def _model_root(self) -> Path | None:
        """Writable model directory, or ``None`` for rapidocr's bundled models.

        rapidocr ships the PP-OCRv6 ``small`` graphs inside the wheel. Those are
        read directly so a default installation never downloads or duplicates
        30 MB of weights. Every other model family is cached next to the
        application (``models/rapidocr``) so packaged builds keep downloads in
        a writable directory.
        """
        if self.spec.det_model_path or self.spec.rec_model_path:
            return None
        if (
            self.spec.det_version == "PP-OCRv6"
            and self.spec.det_model_type == "small"
            and self.spec.rec_version == "PP-OCRv6"
            and self.spec.rec_model_type == "small"
        ):
            return None
        return self.models_dir

    def _engine_params(self, *, use_cuda: bool, use_dml: bool) -> dict[str, Any]:
        """Translate the model spec into the RapidOCR 3.x parameter map."""
        from rapidocr import EngineType, ModelType, OCRVersion

        params: dict[str, Any] = {
            "Global.use_det": True,
            "Global.use_cls": False,
            "Global.use_rec": True,
            "Global.use_preprocess_img": False,
            "Global.text_score": float(self.min_confidence),
            "Global.log_level": "warning",
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.ocr_version": OCRVersion(self.spec.det_version),
            "Det.lang_type": self.spec.det_lang,
            "Det.model_type": ModelType(self.spec.det_model_type),
            "Det.limit_side_len": self.det_limit_side_len,
            "Det.limit_type": "max",
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.ocr_version": OCRVersion(self.spec.rec_version),
            "Rec.lang_type": self.spec.rec_lang,
            "Rec.model_type": ModelType(self.spec.rec_model_type),
            # One batched call per page keeps the GPU busy. Padding waste is
            # still cheaper than waking the session once per detected line.
            "Rec.rec_batch_num": 8 if (use_cuda or use_dml) else 6,
            "EngineConfig.onnxruntime.use_cuda": bool(use_cuda),
            "EngineConfig.onnxruntime.use_dml": bool(use_dml),
        }
        model_root = self._model_root()
        if model_root is not None:
            model_root.mkdir(parents=True, exist_ok=True)
            params["Global.model_root_dir"] = str(model_root)
        if self.spec.det_model_path:
            params["Det.model_path"] = self.spec.det_model_path
        if self.spec.rec_model_path:
            params["Rec.model_path"] = self.spec.rec_model_path
        return params

    def _ensure_model(self) -> None:
        if self._engine is not None:
            return
        with self._lock:
            if self._engine is not None:
                return
            # Must run before rapidocr imports onnxruntime: this puts the CUDA
            # and cuDNN DLL folders shipped by the nvidia-*-cu12 wheels on the
            # Windows search path.
            activate_gpu_runtime()
            try:
                from rapidocr import RapidOCR  # type: ignore[import-not-found]
            except Exception as exc:  # pragma: no cover - optional dependency
                raise OcrUnavailableError(
                    "RapidOCR is not installed. Install `rapidocr` (and an "
                    "onnxruntime variant) or set OTR_OCR_ENGINE=tesseract."
                ) from exc

            attempts: list[tuple[bool, bool]] = []
            if self.use_cuda:
                attempts.append((True, False))
            if self.use_dml:
                attempts.append((False, True))
            attempts.append((False, False))
            last_error: Exception | None = None
            for use_cuda, use_dml in dict.fromkeys(attempts):
                try:
                    logger.info(
                        "Initializing RapidOCR (%s, lang=%s, det=%s/%s, "
                        "rec=%s/%s, cuda=%s, dml=%s)",
                        self.spec.label,
                        self.lang,
                        self.spec.det_version,
                        self.spec.det_model_type,
                        self.spec.rec_version,
                        self.spec.rec_model_type,
                        use_cuda,
                        use_dml,
                    )
                    engine = RapidOCR(
                        params=self._engine_params(
                            use_cuda=use_cuda, use_dml=use_dml
                        )
                    )
                    # Loading the modules eagerly creates the ORT sessions and
                    # downloads any non-bundled weights now, not on the first
                    # screenshot.
                    detector = engine._load_det_model()
                    recognizer = engine._load_rec_model()
                except Exception as exc:  # pragma: no cover - hardware specific
                    last_error = exc
                    if use_cuda or use_dml:
                        logger.warning(
                            "RapidOCR could not initialize on %s (%s); retrying "
                            "with CPUExecutionProvider",
                            "CUDA" if use_cuda else "DirectML",
                            exc,
                        )
                        continue
                    logger.exception("RapidOCR failed to initialize")
                    raise OcrUnavailableError(
                        "RapidOCR could not initialize its ONNX models. Check the "
                        "model download/network logs and the onnxruntime install.",
                        details={"error": str(exc)},
                    ) from exc

                self._engine = engine
                self._detector = detector
                self._recognizer = recognizer
                self.active_providers = self._detect_providers(detector, recognizer)
                self.use_cuda = "CUDAExecutionProvider" in self.active_providers
                self.use_dml = "DmlExecutionProvider" in self.active_providers
                logger.info(
                    "RapidOCR ready (%s) providers=%s",
                    self.spec.label,
                    self.active_providers or self.providers,
                )
                return
            raise OcrUnavailableError(
                "RapidOCR could not initialize its ONNX models.",
                details={"error": str(last_error)},
            ) from last_error

    @staticmethod
    def _detect_providers(detector: Any, recognizer: Any) -> list[str]:
        providers: list[str] = []
        for module in (detector, recognizer):
            session = getattr(getattr(module, "session", None), "session", None)
            get_providers = getattr(session, "get_providers", None)
            if not callable(get_providers):
                continue
            for provider in get_providers():
                if provider not in providers:
                    providers.append(provider)
        return providers

    def _ensure_cpu_model(self) -> None:
        """Build a CPU detector for the fragmented-page retry.

        Only the detector is re-created: detection is the unstable part on the
        GPU, while recognition stays on the fast provider. Loading the 21 MB
        recognizer twice would cost several seconds for no quality gain.
        """
        if self._cpu_detector is not None:
            return
        from rapidocr import RapidOCR  # type: ignore[import-not-found]

        engine = RapidOCR(params=self._engine_params(use_cuda=False, use_dml=False))
        self._cpu_detector = engine._load_det_model()
        self._cpu_engine = engine
        logger.info("CPU OCR detector ready")

    def _read_with_cpu_fallback(
        self, bgr: np.ndarray, image_size: tuple[int, int]
    ) -> list[TextRegion]:
        """Read the page, retrying detection on CPU when boxes look fragmented."""
        regions = self._read_page(bgr, image_size)
        if not (self.cpu_fallback and (self.use_cuda or self.use_dml)):
            return regions
        if not _looks_fragmented(regions):
            return regions
        try:
            self._ensure_cpu_model()
            cpu_regions = self._read_page(
                bgr,
                image_size,
                detector=self._cpu_detector,
                recognizer=self._recognizer,
            )
        except Exception as exc:  # pragma: no cover - hardware specific
            logger.warning("CPU OCR fallback failed: %s", exc)
            return regions
        if not cpu_regions:
            return regions
        fast_score = _fragmentation_score(regions)
        cpu_score = _fragmentation_score(cpu_regions)
        fast_text = sum(len(region.text.strip()) for region in regions)
        cpu_text = sum(len(region.text.strip()) for region in cpu_regions)
        if cpu_score < fast_score or (cpu_score == fast_score and cpu_text > fast_text):
            logger.info(
                "OCR boxes looked fragmented %s; using the CPU detection result %s",
                fast_score,
                cpu_score,
            )
            return cpu_regions
        return regions

    def read(
        self,
        image: Image.Image,
        *,
        refine: bool | None = None,
        **_options: Any,
    ) -> OcrOutput:
        """Run detection + recognition on one page.

        ``refine`` overrides the language default for the zoomed cluster pass.
        A backend whose models match the Japanese profile can therefore serve a
        Japanese job without loading (or running) a second model set.
        """
        started = time.perf_counter()
        self._ensure_model()
        rgb = np.asarray(image.convert("RGB"))
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        with self._lock:
            regions = self._read_with_cpu_fallback(bgr, image.size)
            should_refine = self.lang == "japan" if refine is None else refine
            if should_refine and regions:
                try:
                    regions = self._refine_horizontal_clusters(
                        bgr, image.size, regions
                    )
                except Exception as exc:  # pragma: no cover - model specific
                    logger.warning("Small-text OCR refinement failed: %s", exc)
        regions = deduplicate_regions(regions)
        elapsed = (time.perf_counter() - started) * 1000.0
        logger.info(
            "RapidOCR (%s/%s) found %d regions in %.0f ms (providers=%s)",
            self.lang,
            self.spec.label,
            len(regions),
            elapsed,
            self.active_providers or self.providers,
        )
        return OcrOutput(regions=regions, engine=self.name, elapsed_ms=elapsed)

    def read_language(self, image: Image.Image, *, lang: str) -> OcrOutput:
        """Read a page with the refine behavior of ``lang``.

        The router shares one backend for languages that load identical models;
        this entry point keeps the per-language refinement decision explicit
        instead of mutating ``self.lang`` (which would race across jobs).
        """
        return self.read(image, refine=normalize_ocr_lang(lang) == "japan")

    def refine_regions(
        self, image: Image.Image, regions: list[TextRegion]
    ) -> list[TextRegion]:
        """Run only the zoomed caption/title pass over existing regions.

        Reuses the page's already-recognized text, so a Japanese retry costs a
        handful of small crop detections instead of a second full-page OCR
        pass. Returns ``regions`` unchanged when the backend cannot refine.
        """
        if not regions:
            return regions
        self._ensure_model()
        rgb = np.asarray(image.convert("RGB"))
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        with self._lock:
            try:
                return self._refine_horizontal_clusters(bgr, image.size, regions)
            except Exception as exc:  # pragma: no cover - model specific
                logger.warning("Small-text OCR refinement failed: %s", exc)
                return regions

    def warm_up(self, *, refine: bool = True) -> None:
        """Load the models and run one tiny inference of each kind.

        Called from the server's background warm-up thread. Creating the CUDA
        sessions and loading cuDNN takes several seconds on the first call;
        paying that at startup keeps it off the user's first screenshot.
        """
        self._ensure_model()
        try:
            from rapidocr.ch_ppocr_rec import TextRecInput
        except Exception:  # pragma: no cover - optional dependency
            return
        with self._lock:
            try:
                self._detector(np.full((96, 320, 3), 255, np.uint8))
                self._recognizer(
                    TextRecInput(
                        img=[
                            np.full((48, 192, 3), 255, np.uint8),
                            np.full((48, 192, 3), 255, np.uint8),
                        ],
                        return_word_box=False,
                    )
                )
            except Exception:  # pragma: no cover - hardware/runtime specific
                logger.warning("RapidOCR warm-up inference failed", exc_info=True)

    def _read_page(
        self,
        bgr: np.ndarray,
        image_size: tuple[int, int],
        *,
        detector: Any | None = None,
        recognizer: Any | None = None,
    ) -> list[TextRegion]:
        """Detect line polygons, then recognize all crops in one batch."""
        detector = detector if detector is not None else self._detector
        recognizer = recognizer if recognizer is not None else self._recognizer
        assert detector is not None and recognizer is not None
        try:
            raw = detector(bgr)
        except Exception as exc:
            raise OcrProcessingError(
                "RapidOCR text detection failed.", details={"error": str(exc)}
            ) from exc
        boxes = getattr(raw, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        height, width = bgr.shape[:2]
        rec_scale = self._recognition_scale(image_size)
        crops: list[np.ndarray] = []
        metadata: list[tuple[np.ndarray, bool]] = []
        for box in boxes:
            points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
            if len(points) < 3:
                continue
            points[:, 0] = np.clip(points[:, 0], 0, width)
            points[:, 1] = np.clip(points[:, 1], 0, height)
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            vertical_box = (y1 - y0) > (x1 - x0) * 1.25
            # Vertical columns are tight; extra padding pulls neighbouring
            # background strokes into the crop and corrupts kanji such as 蹂躙.
            pad = 0 if vertical_box else 2
            cx0 = max(0, int(x0) - pad)
            cy0 = max(0, int(y0) - pad)
            cx1 = min(width, int(np.ceil(x1)) + pad)
            cy1 = min(height, int(np.ceil(y1)) + pad)
            crop = bgr[cy0:cy1, cx0:cx1]
            if crop.size == 0:
                continue
            crops.append(self._prepare_crop(crop, rec_scale, vertical_box))
            metadata.append((points, vertical_box))

        recognized = self._recognize_crops(crops, recognizer)
        regions: list[TextRegion] = []
        for (points, vertical), (text, score) in zip(
            metadata, recognized, strict=False
        ):
            text = (text or "").strip()
            if not text or score < self.min_confidence:
                continue
            text = repair_japanese_ocr(text)
            xs = points[:, 0]
            ys = points[:, 1]
            box_w = float(xs.max() - xs.min())
            box_h = float(ys.max() - ys.min())
            polygon = [(float(x), float(y)) for x, y in points]
            if _polygon_area(polygon) < 4:
                continue
            direction = (
                "vertical"
                if vertical
                else detect_text_direction(text, box_w, box_h)
            )
            regions.append(
                TextRegion(
                    id=len(regions),
                    text=text,
                    confidence=round(float(score), 4),
                    polygon=polygon,
                    direction=direction,  # type: ignore[arg-type]
                )
            )
        return regions

    def _prepare_crop(
        self, crop: np.ndarray, rec_scale: float, vertical: bool
    ) -> np.ndarray:
        """Upscale small lettering and rotate vertical columns for the recognizer."""
        if rec_scale > 1.01:
            height, width = crop.shape[:2]
            crop = cv2.resize(
                crop,
                (
                    max(1, round(width * rec_scale)),
                    max(1, round(height * rec_scale)),
                ),
                interpolation=cv2.INTER_CUBIC,
            )
        if vertical and self.rotate_vertical:
            # The recognizer expects left-to-right text. Rotating the column
            # CCW produces exactly that for vertical CJK.
            crop = np.ascontiguousarray(np.rot90(crop, k=1))
        return crop

    def _recognize_crops(
        self, crops: list[np.ndarray], recognizer: Any
    ) -> list[tuple[str, float]]:
        """Recognize a page's crops with one batched RapidOCR call."""
        if not crops:
            return []
        try:
            from rapidocr.ch_ppocr_rec import TextRecInput
        except Exception as exc:  # pragma: no cover - optional dependency
            raise OcrUnavailableError(
                "RapidOCR's recognizer module is unavailable."
            ) from exc
        valid = [
            index
            for index, crop in enumerate(crops)
            if crop.size and crop.shape[0] >= 2 and crop.shape[1] >= 2
        ]
        if not valid:
            return [("", 0.0)] * len(crops)
        # ONNX Runtime's CUDA EP miscomputes this PP-OCR dynamic-width graph on
        # a batch of exactly one (the same crop reads correctly on CPU). Keep
        # every batch at two or more items, and rebalance the tail so a page
        # with e.g. 17 crops never ends with a single-item batch.
        count = len(valid)
        batch_size = min(16, max(2, count))
        if count > batch_size and count % batch_size == 1 and batch_size > 2:
            batch_size -= 1
        with contextlib.suppress(AttributeError):
            recognizer.rec_batch_num = batch_size
        inputs = [crops[index] for index in valid]
        if count == 1:
            inputs = inputs * 2
        try:
            output = recognizer(
                TextRecInput(img=inputs, return_word_box=False)
            )
        except Exception as exc:
            raise OcrProcessingError(
                "RapidOCR text recognition failed.", details={"error": str(exc)}
            ) from exc
        texts = list(getattr(output, "txts", None) or ())
        scores = list(getattr(output, "scores", None) or ())
        results: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        for index, (text, score) in zip(
            valid,
            zip(texts[:count], scores[:count], strict=False),
            strict=False,
        ):
            results[index] = (str(text), float(score))
        return results

    def _refine_horizontal_clusters(
        self,
        bgr: np.ndarray,
        image_size: tuple[int, int],
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        """Re-detect cramped horizontal captions on their own crop.

        The page-level detector sometimes merges a tiny multi-line caption, or
        a title such as ``偽聖女でした``, into one wide box. Cropping just that
        cluster and running detection again at a higher zoom splits it into
        individual lines without paying for a second full-page detection pass.
        """
        refined: list[TextRegion] = []
        for region in regions:
            if region.direction != "horizontal" or not self._needs_cluster_refine(
                region
            ):
                refined.append(region)
                continue
            sub_regions = self._redetect_cluster(bgr, region, image_size)
            if len(sub_regions) >= 2:
                refined.extend(sub_regions)
            else:
                refined.append(region)
        return self._dedupe_spatial(refined)

    @staticmethod
    def _needs_cluster_refine(region: TextRegion) -> bool:
        box = region.bbox
        if box.w < 12 or box.h < 6:
            return False
        # The zoomed retry costs one crop detection (~100-200 ms) per box, so
        # only run it when the OCR result itself looks suspicious. Crisp UI
        # labels are short and wide too (23:48, CHAPTER IV) and never split;
        # merged title/caption fragments arrive with visibly lower scores.
        if len(region.text.strip()) <= 5 and box.w >= box.h * 1.7:
            return region.confidence < 0.92
        if box.h <= 14:
            # Zooming a 9 px caption does not recover missing strokes; it tends
            # to replace a mediocre full-page reading with a worse one. Only
            # retry these when the page-level result is clearly broken.
            return region.confidence < 0.60
        if box.h <= 32:
            return region.confidence < 0.70
        # Large stylized titles are often returned as one garbled line. A
        # zoomed crop re-detects the individual pieces ("偽聖女" + "でした")
        # with near-perfect confidence, which also gives the typesetter the
        # correct baseline.
        return (
            box.h > 32
            and len(region.text.strip()) <= 16
            and region.confidence < 0.85
        )

    @staticmethod
    def _dedupe_spatial(regions: list[TextRegion]) -> list[TextRegion]:
        """Collapse overlapping boxes produced by the zoomed retry.

        The retry can see the same physical line from two neighbouring parent
        crops. Keep the more confident reading (or the longer one when the
        scores are close) and drop the duplicate. IoU is used instead of
        one-sided containment: a small caption printed over a large title is
        physically inside the title box but is not a duplicate of it.
        """
        kept: list[TextRegion] = []
        for region in regions:
            duplicate_index = None
            for index, other in enumerate(kept):
                ba, bb = region.bbox, other.bbox
                x0, y0 = max(ba.x, bb.x), max(ba.y, bb.y)
                x1, y1 = min(ba.right, bb.right), min(ba.bottom, bb.bottom)
                inter = max(0, x1 - x0) * max(0, y1 - y0)
                if inter <= 0:
                    continue
                union = ba.area + bb.area - inter
                if union <= 0 or inter / union < 0.5:
                    continue
                duplicate_index = index
                break
            if duplicate_index is None:
                kept.append(region)
                continue
            other = kept[duplicate_index]
            replace = region.confidence > other.confidence + 0.08 or (
                abs(region.confidence - other.confidence) <= 0.08
                and len(region.text) > len(other.text)
            )
            if replace:
                kept[duplicate_index] = region
        return kept

    def _redetect_cluster(
        self,
        bgr: np.ndarray,
        region: TextRegion,
        image_size: tuple[int, int],
    ) -> list[TextRegion]:
        detector = self._detector
        recognizer = self._recognizer
        if detector is None or recognizer is None:
            return []
        box = region.bbox.clamp(*image_size)
        # Caption ribbons are stacked tightly: a 3-line caption is often
        # detected as only its lower two lines. Pull in enough context above the
        # box to see the missed first line.
        pad = (
            max(6, int(round(box.h * 1.2)))
            if box.h <= 22
            else max(4, int(round(box.h * 0.35)))
        )
        center_tolerance = max(4, int(round(box.h * 0.7)))
        x0 = max(0, box.x - pad)
        y0 = max(0, box.y - pad)
        x1 = min(bgr.shape[1], box.right + pad)
        y1 = min(bgr.shape[0], box.bottom + pad)
        if x1 - x0 < 12 or y1 - y0 < 8:
            return []
        crop = bgr[y0:y1, x0:x1]
        zoom = min(4.5, 900.0 / max(1, max(crop.shape[:2])))
        zoom = max(1.0, zoom)
        if zoom > 1.01:
            crop = cv2.resize(
                crop,
                (
                    max(1, round(crop.shape[1] * zoom)),
                    max(1, round(crop.shape[0] * zoom)),
                ),
                interpolation=cv2.INTER_CUBIC,
            )
        try:
            raw = detector(crop)
        except Exception:
            return []
        boxes = getattr(raw, "boxes", None)
        if boxes is None:
            return []
        sub_crops: list[np.ndarray] = []
        metadata: list[tuple[np.ndarray, bool]] = []
        # A large title crop also contains the tiny caption printed beside it.
        # Keep only pieces that are tall enough to belong to the parent line;
        # otherwise the zoomed pass would replace a clean title with caption
        # fragments read at 9 px.
        min_piece_height = max(6.0, box.h * 0.22)
        for polygon in boxes:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(points) < 3:
                continue
            px0, py0 = points.min(axis=0)
            px1, py1 = points.max(axis=0)
            if px1 - px0 < 6 or py1 - py0 < 4:
                continue
            if (py1 - py0) / zoom < min_piece_height:
                continue
            center_x = x0 + (px0 + px1) / 2.0 / zoom
            center_y = y0 + (py0 + py1) / 2.0 / zoom
            if not (box.x - 4 <= center_x <= box.right + 4):
                continue
            if not (
                box.y - center_tolerance
                <= center_y
                <= box.bottom + center_tolerance
            ):
                continue
            scx0 = max(0, int(px0) - 2)
            scy0 = max(0, int(py0) - 2)
            scx1 = min(crop.shape[1], int(px1) + 3)
            scy1 = min(crop.shape[0], int(py1) + 3)
            sub_crop = crop[scy0:scy1, scx0:scx1]
            if sub_crop.size == 0:
                continue
            sub_h, sub_w = sub_crop.shape[:2]
            vertical = sub_h > sub_w * 1.25
            sub_crops.append(self._prepare_crop(sub_crop, 1.0, vertical))
            metadata.append((points, vertical))

        recognized = self._recognize_crops(sub_crops, recognizer)
        sub_regions: list[TextRegion] = []
        for (points, vertical), (text, score) in zip(
            metadata, recognized, strict=False
        ):
            text = repair_japanese_ocr((text or "").strip())
            if (
                not text
                or score < (0.25 if box.h <= 22 else 0.30)
                or not _meaningful_text(text)
            ):
                continue
            source_points = [
                (
                    min(max(0.0, x0 + float(px) / zoom), float(image_size[0])),
                    min(max(0.0, y0 + float(py) / zoom), float(image_size[1])),
                )
                for px, py in points
            ]
            source_box = TextRegion(
                id=0, text=text, polygon=source_points
            ).bbox
            if source_box.area < 40:
                continue
            sub_regions.append(
                TextRegion(
                    id=0,
                    text=text,
                    confidence=round(float(score), 4),
                    polygon=source_points,
                    direction="vertical" if vertical else "horizontal",
                )
            )
        if len(sub_regions) < 2:
            return []
        combined = "".join(item.text for item in sub_regions)
        if len(combined) < max(2, int(len(region.text) * 0.6)):
            return []
        return sub_regions

    @staticmethod
    def _recognition_scale(size: tuple[int, int]) -> float:
        """Upscale factor applied only to recognition crops.

        RapidOCR's detector is robust at native resolution; the recognizer is
        what loses 6-12 px manga lettering. Keeping detection at 1x avoids the
        quadratic cost of upscaling the whole page.
        """
        width, height = size
        long = max(width, height)
        scale = min(2.5, 3200.0 / max(1, long))
        if scale < 1.1:
            return 1.0
        return round(scale, 3)


class LanguageAwareOCRBackend:
    """Route each job to the RapidOCR language/model that fits it.

    ``source_lang`` selects the recognizer explicitly. In ``auto`` mode Japanese
    manga is routed to the Japanese profile, and pages whose fast pass contains
    kana (or almost no usable text) are retried with the Japanese recognizer.
    Model instances are created lazily and cached per language.
    """

    name = "rapidocr"

    def __init__(
        self,
        factory: Any,
        *,
        default_lang: str = "ch",
        auto_fallback: bool = True,
        cache_size: int = 8,
    ) -> None:
        self._factory = factory
        self.default_lang = normalize_ocr_lang(default_lang)
        self.auto_fallback = auto_fallback
        self._backends: dict[str, OCRBackend] = {}
        self._lock = threading.Lock()
        # Re-translating the same screenshot (the web UI and the desktop HUD can
        # both resubmit an unchanged frame) may reuse the OCR result. The key is
        # an exact digest, so a changed pixel always misses.
        self._cache_size = max(0, int(cache_size))
        self._page_cache: OrderedDict[tuple[Any, ...], OcrOutput] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _backend(self, lang: str) -> OCRBackend:
        key = normalize_ocr_lang(lang, self.default_lang)
        with self._lock:
            backend = self._backends.get(key)
            if backend is not None:
                return backend
            candidate = self._factory(key)
            # Languages that resolve to the same detector/recognizer share one
            # initialized backend. With PP-OCRv6, ``ch`` and ``japan`` use the
            # same multilingual graphs, so a Japanese page never pays a second
            # model initialization cost.
            signature = getattr(candidate, "model_signature", None)
            if callable(signature):
                wanted = signature()
                for shared in self._backends.values():
                    other = getattr(shared, "model_signature", None)
                    if callable(other) and other() == wanted:
                        candidate = shared
                        break
            self._backends[key] = candidate
            return candidate

    def warm_up(self) -> None:
        """Initialize the models in the background at server startup."""
        languages = [self.default_lang]
        if self.auto_fallback and self.default_lang in {"ch", "japan", "chinese_cht"}:
            languages.append("japan")
        warmed: set[int] = set()
        for lang in dict.fromkeys(languages):
            try:
                backend = self._backend(lang)
            except Exception:  # pragma: no cover - install specific
                logger.warning("OCR warm-up could not build %s", lang, exc_info=True)
                continue
            if id(backend) in warmed:
                continue
            warm = getattr(backend, "warm_up", None)
            if not callable(warm):
                continue
            warmed.add(id(backend))
            try:
                warm(refine=lang == "japan")
            except TypeError:
                warm()
            except Exception:  # pragma: no cover - hardware specific
                logger.warning("OCR warm-up failed for %s", lang, exc_info=True)

    def _resolve(self, source_lang: str, mode: str) -> tuple[str, bool]:
        lang = (source_lang or "auto").strip().lower()
        if lang == "auto":
            if mode == "manga":
                return "japan", True
            return self.default_lang, False
        normalized = normalize_ocr_lang(lang, self.default_lang)
        if normalized == "ch" and lang in {"zh", "zh-cn", "cn"}:
            return "ch", True
        return normalized, True

    @staticmethod
    def _read_as(backend: OCRBackend, image: Image.Image, lang: str) -> OcrOutput:
        reader = getattr(backend, "read_language", None)
        if callable(reader):
            return reader(image, lang=lang)
        return backend.read(image)

    def _refine_in_place(
        self, backend: OCRBackend, image: Image.Image, output: OcrOutput
    ) -> OcrOutput | None:
        """Refine an already-recognized Japanese page without a second pass."""
        refine = getattr(backend, "refine_regions", None)
        if not callable(refine):
            return None
        same_models = getattr(backend, "can_refine_as_japanese", None)
        if callable(same_models) and not same_models():
            return None
        if not output.regions:
            # The Japanese pass would load the same weights and see the same
            # empty page. Keep the result and skip the retry entirely.
            return output
        try:
            regions = refine(image, output.regions)
        except Exception as exc:  # pragma: no cover - model specific
            logger.warning("In-place Japanese refinement failed: %s", exc)
            return None
        if regions is output.regions or regions == output.regions:
            return output
        refined = OcrOutput(
            regions=regions,
            engine=output.engine,
            elapsed_ms=output.elapsed_ms,
            warnings=list(output.warnings),
        )
        refined.warnings.append(
            "Auto-detected Japanese text and refined cramped lines in place "
            "(no second OCR pass)."
        )
        return refined

    @staticmethod
    def _clone_output(output: OcrOutput, *, elapsed_ms: float | None = None) -> OcrOutput:
        return OcrOutput(
            regions=[region.model_copy(deep=True) for region in output.regions],
            engine=output.engine,
            elapsed_ms=output.elapsed_ms if elapsed_ms is None else elapsed_ms,
            warnings=list(output.warnings),
        )

    @staticmethod
    def _page_key(
        image: Image.Image, lang: str, explicit: bool
    ) -> tuple[Any, ...]:
        digest = hashlib.blake2b(image.tobytes(), digest_size=16).digest()
        return (digest, image.width, image.height, image.mode, lang, explicit)

    def _cache_lookup(self, key: tuple[Any, ...]) -> OcrOutput | None:
        with self._cache_lock:
            cached = self._page_cache.get(key)
            if cached is None:
                return None
            self._page_cache.move_to_end(key)
            return cached

    def _cache_store(self, key: tuple[Any, ...], output: OcrOutput) -> None:
        stored = self._clone_output(output)
        with self._cache_lock:
            self._page_cache[key] = stored
            self._page_cache.move_to_end(key)
            while len(self._page_cache) > self._cache_size:
                self._page_cache.popitem(last=False)

    def read(self, image: Image.Image, **options: Any) -> OcrOutput:
        source_lang = str(options.get("source_lang") or "auto")
        mode = str(options.get("mode") or "auto")
        lang, explicit = self._resolve(source_lang, mode)
        key: tuple[Any, ...] | None = None
        if self._cache_size > 0 and isinstance(image, Image.Image):
            key = self._page_key(image, lang, explicit)
            cached = self._cache_lookup(key)
            if cached is not None:
                logger.debug("OCR page cache hit (%s)", lang)
                return self._clone_output(cached, elapsed_ms=0.0)
        output = self._read_uncached(image, lang, explicit)
        if key is not None:
            self._cache_store(key, output)
        return output

    def _read_uncached(
        self, image: Image.Image, lang: str, explicit: bool
    ) -> OcrOutput:
        backend = self._backend(lang)
        output = self._read_as(backend, image, lang)
        # Keep the public engine name stable for older clients, but make the
        # selected recognizer visible in diagnostics.
        if output.warnings is None:
            output.warnings = []
        if (
            not explicit
            and lang != "japan"
            and self.auto_fallback
            and _needs_japanese_retry(output)
        ):
            refined = self._refine_in_place(backend, image, output)
            if refined is not None:
                return refined
            try:
                japan = self._backend("japan")
                alternative = self._read_as(japan, image, "japan")
            except Exception as exc:  # pragma: no cover - install specific
                output.warnings.append(
                    f"Japanese OCR retry failed: {exc}"
                )
                return output
            fast_text = "".join(region.text for region in output.regions)
            alt_text = "".join(region.text for region in alternative.regions)
            if alternative.regions and (
                contains_kana(fast_text)
                or _ocr_quality(alternative) > _ocr_quality(output) * 1.05
            ):
                alternative.warnings.append(
                    "Auto-detected Japanese text and used the Japanese OCR model."
                )
                return alternative
            if alternative.regions and contains_kana(alt_text):
                alternative.warnings.append(
                    "Japanese OCR model produced kana; prefer it over the "
                    "language-neutral model."
                )
                return alternative
        return output


def _needs_japanese_retry(output: OcrOutput) -> bool:
    regions = output.regions
    if not regions:
        return True
    text = "".join(region.text for region in regions)
    if contains_kana(text):
        return True
    vertical = sum(1 for region in regions if region.direction == "vertical")
    if regions and vertical / len(regions) >= 0.5:
        return True
    meaningful = sum(1 for ch in text if ch.isalnum() or contains_cjk(ch))
    return meaningful <= 3


def _ocr_quality(output: OcrOutput) -> float:
    score = 0.0
    for region in output.regions:
        text = region.text.strip()
        score += len(text) * max(0.05, region.confidence)
        if contains_kana(text):
            score += 2.0
    return score + len(output.regions) * 1.5





class TesseractBackend:
    name = "tesseract"

    def __init__(
        self,
        *,
        lang: str = "chi_sim+eng",
        min_confidence: float = 0.5,
        cmd: str | None = None,
    ) -> None:
        self.lang = lang
        self.min_confidence = min_confidence
        self.cmd = cmd
        self._lock = threading.Lock()

    @staticmethod
    def is_available() -> bool:
        return bool(importlib.util.find_spec("pytesseract")) and bool(
            importlib.util.find_spec("PIL")
        )

    def read(self, image: Image.Image, **_options: Any) -> OcrOutput:
        started = time.perf_counter()
        try:
            import pytesseract
            from pytesseract import Output
        except Exception as exc:  # pragma: no cover - environment specific
            raise OcrUnavailableError(
                "pytesseract is not installed. Install it plus the Tesseract binary."
            ) from exc
        if self.cmd:
            pytesseract.pytesseract.tesseract_cmd = self.cmd
        try:
            with self._lock:
                data = pytesseract.image_to_data(
                    image.convert("RGB"),
                    lang=self.lang,
                    output_type=Output.DICT,
                )
        except Exception as exc:
            raise OcrProcessingError(
                "Tesseract OCR failed. Is the native tesseract binary on PATH?",
                details={"error": str(exc)},
            ) from exc

        groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        count = len(data.get("text", []))
        for index in range(count):
            text = (data["text"][index] or "").strip()
            if not text:
                continue
            try:
                confidence = float(data["conf"][index]) / 100.0
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < self.min_confidence:
                continue
            key = (
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
            )
            groups[key].append(index)

        regions: list[TextRegion] = []
        for indices in groups.values():
            words = [(data["text"][i] or "").strip() for i in indices]
            text = " ".join(w for w in words if w)
            if not text:
                continue
            left = min(int(data["left"][i]) for i in indices)
            top = min(int(data["top"][i]) for i in indices)
            right = max(int(data["left"][i]) + int(data["width"][i]) for i in indices)
            bottom = max(int(data["top"][i]) + int(data["height"][i]) for i in indices)
            confidences = [float(data["conf"][i]) / 100.0 for i in indices]
            direction = detect_text_direction(text, right - left, bottom - top)
            regions.append(
                TextRegion(
                    id=len(regions),
                    text=text,
                    confidence=round(float(np.mean(confidences)), 4),
                    polygon=[
                        (float(left), float(top)),
                        (float(right), float(top)),
                        (float(right), float(bottom)),
                        (float(left), float(bottom)),
                    ],
                    direction=direction,  # type: ignore[arg-type]
                )
            )
        elapsed = (time.perf_counter() - started) * 1000.0
        return OcrOutput(regions=regions, engine=self.name, elapsed_ms=elapsed)


class DisabledOCRBackend:
    name = "none"

    def read(self, image: Image.Image, **_options: Any) -> OcrOutput:
        raise OcrUnavailableError(
            "OCR is disabled. Set OTR_OCR_ENGINE=rapidocr or tesseract."
        )


class AutoOCRBackend:
    """Try RapidOCR first, then Tesseract, preserving a helpful warning trail."""

    name = "auto"

    def __init__(self, backends: list[OCRBackend]) -> None:
        if not backends:
            raise OcrUnavailableError("No OCR backend is configured.")
        self.backends = backends
        self._active: OCRBackend | None = None

    def warm_up(self) -> None:
        """Initialize the first usable backend in the background."""
        for backend in self.backends:
            warm = getattr(backend, "warm_up", None)
            if not callable(warm):
                continue
            try:
                warm()
                return
            except Exception:  # pragma: no cover - install specific
                logger.warning(
                    "OCR warm-up failed for %s", backend.name, exc_info=True
                )

    def read(self, image: Image.Image, **options: Any) -> OcrOutput:
        warnings: list[str] = []
        candidates = list(self.backends)
        if self._active in candidates:
            candidates.remove(self._active)  # type: ignore[arg-type]
            candidates.insert(0, self._active)  # type: ignore[arg-type]
        last_error: Exception | None = None
        last_empty: OcrOutput | None = None
        for backend in candidates:
            try:
                output = read_with_options(backend, image, options)
                if output.regions:
                    self._active = backend
                    output.warnings = warnings + output.warnings
                    return output
                warnings.append(f"{backend.name}: no text detected")
                last_empty = output
            except OTranslatorError as exc:
                warnings.append(f"{backend.name}: {exc.user_message}")
                last_error = exc
                logger.warning("OCR backend %s failed: %s", backend.name, exc)
        if last_empty is not None:
            last_empty.warnings = warnings
            return last_empty
        if isinstance(last_error, OcrUnavailableError):
            raise last_error
        raise OcrProcessingError(
            "All OCR backends failed.", details={"warnings": warnings}
        )


def read_with_options(
    backend: OCRBackend, image: Image.Image, options: dict[str, Any]
) -> OcrOutput:
    """Pass job language hints to backends that understand them.

    Third-party or test backends may still implement the original single-argument
    ``read`` contract; fall back to that without hiding real errors.
    """
    try:
        return backend.read(image, **options)
    except TypeError as exc:
        if "unexpected keyword" not in str(exc):
            raise
        return backend.read(image)


def build_ocr_backend(
    *,
    engine: str = "auto",
    ocr_lang: str = "ch",
    ocr_version: str = "auto",
    ocr_profile: str = "auto",
    models_dir: str | Path | None = None,
    det_model_path: str | None = None,
    rec_model_path: str | None = None,
    det_limit_side_len: int = 960,
    auto_fallback: bool = True,
    min_confidence: float = 0.5,
    tesseract_lang: str = "chi_sim+eng",
    tesseract_cmd: str | None = None,
    device: str = "auto",
    providers: str | None = None,
    cache_size: int = 8,
) -> OCRBackend:
    """Construct the configured OCR chain without loading model weights."""

    def _rapidocr_factory(lang: str) -> OCRBackend:
        return RapidOCRBackend(
            lang=lang,
            version=ocr_version,
            profile=ocr_profile,
            models_dir=models_dir,
            det_model_path=det_model_path,
            rec_model_path=rec_model_path,
            det_limit_side_len=det_limit_side_len,
            min_confidence=min_confidence,
            device=device,
            providers=providers,
        )

    normalized_engine = (engine or "auto").strip().lower()
    # ``paddle`` and ``onnx`` are accepted as aliases so existing .env files
    # keep starting after the RapidOCR migration.
    if normalized_engine in {"paddle", "onnx"}:
        if normalized_engine == "paddle":
            logger.warning(
                "OTR_OCR_ENGINE=paddle is deprecated; RapidOCR (ONNX) is now "
                "the OCR engine."
            )
        normalized_engine = "rapidocr"
    if normalized_engine == "none":
        return DisabledOCRBackend()
    if normalized_engine == "rapidocr":
        if not RapidOCRBackend.is_available():
            raise OcrUnavailableError(
                "RapidOCR is not installed. Run `pip install rapidocr` (or "
                "`pip install -r requirements.txt`) and restart the server."
            )
        return LanguageAwareOCRBackend(
            _rapidocr_factory,
            default_lang=ocr_lang,
            auto_fallback=auto_fallback,
            cache_size=cache_size,
        )
    if normalized_engine == "tesseract":
        return TesseractBackend(
            lang=tesseract_lang,
            min_confidence=min_confidence,
            cmd=tesseract_cmd,
        )

    candidates: list[OCRBackend] = []
    if RapidOCRBackend.is_available():
        candidates.append(
            LanguageAwareOCRBackend(
                _rapidocr_factory,
                default_lang=ocr_lang,
                auto_fallback=auto_fallback,
                cache_size=cache_size,
            )
        )
    if TesseractBackend.is_available():
        candidates.append(
            TesseractBackend(
                lang=tesseract_lang,
                min_confidence=min_confidence,
                cmd=tesseract_cmd,
            )
        )
    if not candidates:
        raise OcrUnavailableError(
            "No OCR engine is available. Install RapidOCR (`pip install "
            "rapidocr`) or pytesseract + Tesseract, then restart the server."
        )
    return AutoOCRBackend(candidates)


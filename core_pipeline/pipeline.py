"""End-to-end orchestration of OCR, inpainting, translation, and rendering."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from PIL import Image

from core_pipeline.errors import OTranslatorError
from core_pipeline.image_utils import decode_image
from core_pipeline.inpainting import Inpainter
from core_pipeline.mask import build_mask, mask_coverage, mask_to_pil
from core_pipeline.ocr import (
    OCRBackend,
    classify_region_roles,
    contains_kana,
    merge_line_fragments,
    merge_paragraph_lines,
    merge_stacked_lines,
    merge_title_pieces,
    order_regions,
    read_with_options,
    resolve_overlapping_regions,
)
from core_pipeline.translation import Translator
from core_pipeline.types import (
    PipelineOptions,
    PipelineResult,
    RenderBlock,
    TextRegion,
    TranslationOutput,
)
from core_pipeline.typesetting import Renderer

logger = logging.getLogger(__name__)

# OCR often drops the space between a handle and its relative timestamp
# ("ai_cosmic_horror_art4小时"). Metadata is not translated, so restore the
# visual gap before it is redrawn.
_METADATA_TIME_GAP_RE = re.compile(
    r"(?<=[A-Za-z0-9._])(?=(?:"
    r"\d+\s*(?:秒|秒钟|分钟|分|小时|時間|时间|天|日|周|週|星期|个月|月|年)"
    r"|\d+\s*(?:s|m|h|d|w|y)(?![A-Za-z])"
    r"|\d+\s*[^\x00-\x7f]"
    r"))",
    re.IGNORECASE,
)


# PP-OCR occasionally reads the CJK hour unit as a Cyrillic capital Es plus a
# modifier letter when the handle immediately precedes it. Since metadata is
# not sent to the LLM, repair that frequent confusion locally.
_METADATA_HOUR_OCR_RE = re.compile(r"(?<=\d)\s*(?:Сʱ|сʱ|СЧ|сч)")


def _display_metadata(text: str, target_lang: str = "zh") -> str:
    value = (text or "").strip()
    if target_lang.lower() in {"zh", "zh-cn", "zh-hans", "chinese"}:
        value = _METADATA_HOUR_OCR_RE.sub("小时", value)
    elif target_lang.lower() in {"zh-tw", "zh-hant"}:
        value = _METADATA_HOUR_OCR_RE.sub("小時", value)
    return _METADATA_TIME_GAP_RE.sub(" ", value).strip()


ProgressCallback = Callable[[str, int, str], None]
PreviewCallback = Callable[
    [Image.Image, list[RenderBlock], list[TextRegion], dict[int, str], dict[str, float]],
    Awaitable[None],
]


class TranslationPipeline:
    def __init__(
        self,
        *,
        ocr: OCRBackend,
        inpainter: Inpainter,
        translator: Translator,
        renderer: Renderer,
        max_image_side: int = 2400,
        max_image_pixels: int = 40_000_000,
        dilate_px: int = 7,
        translation_fallback_to_source: bool = True,
        progressive_preview: bool = True,
    ) -> None:
        self.ocr = ocr
        self.inpainter = inpainter
        self.translator = translator
        self.renderer = renderer
        self.max_image_side = max_image_side
        self.max_image_pixels = max_image_pixels
        self.dilate_px = dilate_px
        self.translation_fallback_to_source = translation_fallback_to_source
        self.progressive_preview = progressive_preview

    @staticmethod
    def _report(
        callback: ProgressCallback | None, stage: str, progress: int, message: str
    ) -> None:
        if callback is None:
            return
        try:
            callback(stage, max(0, min(100, progress)), message)
        except Exception:  # progress reporting must never break the pipeline
            logger.warning("Progress callback failed", exc_info=True)

    async def run(
        self,
        image_bytes: bytes,
        options: PipelineOptions,
        *,
        job_id: str | None = None,
        progress: ProgressCallback | None = None,
        on_preview: PreviewCallback | None = None,
    ) -> PipelineResult:
        job_id = job_id or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        timings: dict[str, float] = {}
        warnings: list[str] = []

        self._report(progress, "decoding", 5, "Decoding and normalizing image")
        decode_started = time.perf_counter()
        original, source_size = await asyncio.to_thread(
            decode_image,
            image_bytes,
            max_side=self.max_image_side,
            max_pixels=self.max_image_pixels,
        )
        timings["decode_ms"] = (time.perf_counter() - decode_started) * 1000.0
        processed_size = original.size

        self._report(progress, "ocr", 15, "Detecting text with OCR")
        ocr_started = time.perf_counter()
        ocr_output = await asyncio.to_thread(
            read_with_options,
            self.ocr,
            original,
            {"source_lang": options.source_lang, "mode": options.mode},
        )
        timings["ocr_ms"] = (time.perf_counter() - ocr_started) * 1000.0
        warnings.extend(ocr_output.warnings)
        regions = order_regions(
            ocr_output.regions,
            mode=options.mode,
            reading_order=options.reading_order,
            source_lang=options.source_lang,
        )
        # Separate usernames/timestamps and tag-only rows before paragraph
        # reconstruction; otherwise a social header is welded onto its caption.
        regions = classify_region_roles(regions)
        # Fragmented detection (common on small, low-contrast captures) returns
        # several boxes for one physical line. Join those before the vertical
        # passes; otherwise the stacked/paragraph mergers build tall regions
        # from fragments of different columns and two translations overprint.
        regions = merge_line_fragments(regions)
        merge_layout = options.mode in {"manga", "novel"} or (
            options.mode == "auto"
            and any(contains_kana(region.text) for region in regions)
        )
        if merge_layout:
            # Tiny stacked subtitle/caption lines are one sentence; merging
            # them keeps the translation coherent and the layout intact.
            regions = merge_stacked_lines(regions)
            # Same-baseline title pieces (偽聖女 + でした) are one sentence too.
            regions = merge_title_pieces(regions)
        if options.mode != "ui":
            # Reconstruct wrapped paragraphs (social posts, chat messages,
            # novel captions) so the translator receives a whole sentence
            # instead of one request item per physical OCR line.
            regions = merge_paragraph_lines(
                regions, page_width=processed_size[0]
            )
        # Last-resort layout guard: any two text regions still covering the
        # same pixels must become one block, or their translations are painted
        # on top of each other.
        regions = resolve_overlapping_regions(regions)

        if not regions:
            self._report(progress, "complete", 100, "No text detected")
            timings["total_ms"] = (time.perf_counter() - started) * 1000.0
            empty_mask = build_mask(processed_size, [], dilate_px=0)
            return PipelineResult(
                job_id=job_id,
                original=original,
                clean=original.copy(),
                mask=mask_to_pil(empty_mask),
                rendered=original.copy(),
                regions=[],
                translations={},
                blocks=[],
                ocr_engine=ocr_output.engine,
                inpaint_engine="not-run",
                translation_provider=self.translator.name,
                translation_model=self.translator.model,
                degraded=False,
                warnings=[*warnings, "No text was detected in this screenshot."],
                timings_ms=timings,
                options=options,
                source_size=source_size,
                processed_size=processed_size,
                empty=True,
            )

        self._report(progress, "masking", 42, "Building and dilating text mask")
        mask_started = time.perf_counter()
        dilate_px = options.dilate_px if options.dilate_px is not None else self.dilate_px
        mask = await asyncio.to_thread(
            build_mask,
            processed_size,
            regions,
            dilate_px=dilate_px,
        )
        timings["mask_ms"] = (time.perf_counter() - mask_started) * 1000.0
        logger.info(
            "Mask covers %.1f%% of %dx%d image",
            mask_coverage(mask) * 100,
            processed_size[0],
            processed_size[1],
        )

        self._report(
            progress,
            "processing",
            55,
            "Inpainting background and translating text in parallel",
        )

        # Cheap first pass: local Telea/median fills. It doubles as the
        # progressive-preview background while LaMa refines textured ROIs.
        fast = None
        fast_pass = getattr(self.inpainter, "fast_pass", None)
        if callable(fast_pass):
            try:
                fast_started = time.perf_counter()
                fast = await asyncio.to_thread(fast_pass, original, mask, regions)
                timings["inpaint_fast_ms"] = (
                    time.perf_counter() - fast_started
                ) * 1000.0
            except Exception:  # never let the preview path break the job
                logger.warning("Fast inpainting pass failed", exc_info=True)

        async def run_inpaint():
            with_fast = getattr(self.inpainter, "inpaint_with_fast", None)
            if fast is not None and callable(with_fast):
                output = await asyncio.to_thread(
                    with_fast, original, mask, regions, fast
                )
            else:
                output = await asyncio.to_thread(self.inpainter.inpaint, original, mask)
            self._report(progress, "inpainting", 72, "Background inpainted")
            return output

        # Usernames, timestamps and action labels are chrome: translating them
        # is both unnecessary and how they end up welded onto the caption. Keep
        # them verbatim (with the OCR-dropped gap restored) and only send the
        # post's content to the LLM. Counters (``18``, ``8.2万``) are chrome
        # too, and must not even enter the mask: their OCR box usually also
        # contains the neighbouring repost/like icon.
        metadata_translations = {
            region.id: _display_metadata(region.text, options.target_lang)
            for region in regions
            if region.role == "metadata"
        }
        keep_translations = {
            region.id: region.text for region in regions if region.role == "keep"
        }
        source_translations = {
            region.id: keep_translations.get(
                region.id,
                metadata_translations.get(region.id, region.text),
            )
            for region in regions
        }
        translatable = [
            region
            for region in regions
            if region.role not in {"metadata", "keep"}
        ]

        async def run_translation() -> TranslationOutput:
            if not options.translate:
                return TranslationOutput(
                    translations=dict(source_translations),
                    provider="disabled",
                    model="none",
                    elapsed_ms=0.0,
                    degraded=True,
                    warnings=["Translation was disabled for this job."],
                )
            if options.source_lang == options.target_lang:
                return TranslationOutput(
                    translations=dict(source_translations),
                    provider="skipped",
                    model="none",
                    elapsed_ms=0.0,
                    degraded=True,
                    warnings=[
                        "Source and target languages are identical; translation was skipped."
                    ],
                )
            if not translatable:
                return TranslationOutput(
                    translations={
                        **metadata_translations,
                        **keep_translations,
                    },
                    provider=self.translator.name,
                    model=self.translator.model,
                    elapsed_ms=0.0,
                )
            try:
                output = await self.translator.translate(translatable, options)
                output.translations.update(metadata_translations)
                output.translations.update(keep_translations)
                self._report(progress, "translation", 72, "Translation complete")
                return output
            except Exception as exc:
                if not self.translation_fallback_to_source:
                    raise
                user_message = (
                    exc.user_message
                    if isinstance(exc, OTranslatorError)
                    else f"Translation failed: {exc}"
                )
                logger.exception("Translation failed; falling back to source text")
                return TranslationOutput(
                    translations=dict(source_translations),
                    provider="fallback",
                    model=self.translator.model,
                    elapsed_ms=0.0,
                    degraded=True,
                    warnings=[
                        user_message,
                        "Source text was kept so the page could still be rendered.",
                    ],
                )

        translation_task = asyncio.create_task(run_translation())
        inpaint_task = asyncio.create_task(run_inpaint())
        try:
            translation_output = await translation_task
        except Exception:
            inpaint_task.cancel()
            await asyncio.gather(inpaint_task, return_exceptions=True)
            raise

        # Pass 1: text + solid/fast-filled background, before LaMa finishes.
        if (
            self.progressive_preview
            and on_preview is not None
            and fast is not None
        ):
            preview_started = time.perf_counter()
            preview_render = await asyncio.to_thread(
                self.renderer.render,
                original,
                fast.image,
                regions,
                translation_output.translations,
                target_lang=options.target_lang,
                font_scale=options.font_scale,
                mode=options.mode,
            )
            timings["preview_render_ms"] = (
                time.perf_counter() - preview_started
            ) * 1000.0
            timings["preview_ready_ms"] = (time.perf_counter() - started) * 1000.0
            self._report(progress, "preview", 82, "Preview ready; restoring background")
            try:
                await on_preview(
                    preview_render.image,
                    preview_render.blocks,
                    regions,
                    translation_output.translations,
                    {
                        key: round(value, 1)
                        for key, value in timings.items()
                    },
                )
            except Exception:  # a failed preview must not fail the job
                logger.warning("Progressive preview callback failed", exc_info=True)

        inpaint_output = await inpaint_task
        timings["inpaint_ms"] = inpaint_output.elapsed_ms
        timings["inpaint_roi_ms"] = max(
            0.0, inpaint_output.elapsed_ms - timings.get("inpaint_fast_ms", 0.0)
        )
        timings["inpaint_fast_regions"] = float(inpaint_output.fast_regions)
        timings["inpaint_lama_regions"] = float(inpaint_output.lama_regions)
        timings["translation_ms"] = translation_output.elapsed_ms
        warnings.extend(inpaint_output.warnings)
        warnings.extend(translation_output.warnings)

        self._report(progress, "rendering", 88, "Typesetting translated text")
        render_started = time.perf_counter()
        render_output = await asyncio.to_thread(
            self.renderer.render,
            original,
            inpaint_output.image,
            regions,
            translation_output.translations,
            target_lang=options.target_lang,
            font_scale=options.font_scale,
            mode=options.mode,
        )
        timings["render_ms"] = (time.perf_counter() - render_started) * 1000.0
        warnings.extend(render_output.warnings)
        timings["total_ms"] = (time.perf_counter() - started) * 1000.0

        self._report(progress, "complete", 100, "Translation complete")
        degraded = translation_output.degraded or bool(
            translation_output.provider in {"fallback", "echo", "disabled"}
        )
        logger.info(
            "Job %s: %d regions, OCR=%.0fms inpaint=%.0fms (fast=%.0fms, "
            "roi_patches=%d) translation=%.0fms render=%.0fms",
            job_id,
            len(regions),
            timings.get("ocr_ms", 0),
            timings.get("inpaint_ms", 0),
            timings.get("inpaint_fast_ms", 0),
            inpaint_output.roi_patches,
            timings.get("translation_ms", 0),
            timings.get("render_ms", 0),
        )
        return PipelineResult(
            job_id=job_id,
            original=original,
            clean=inpaint_output.image,
            mask=mask_to_pil(mask),
            rendered=render_output.image,
            regions=regions,
            translations=translation_output.translations,
            blocks=render_output.blocks,
            ocr_engine=ocr_output.engine,
            inpaint_engine=inpaint_output.engine,
            translation_provider=translation_output.provider,
            translation_model=translation_output.model,
            degraded=degraded,
            warnings=_dedupe(warnings),
            timings_ms={key: round(value, 1) for key, value in timings.items()},
            options=options,
            source_size=source_size,
            processed_size=processed_size,
            translation_usage=translation_output.usage,
            inpaint_strategy=inpaint_output.strategy,
            empty=False,
        )

    async def aclose(self) -> None:
        await self.translator.aclose()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
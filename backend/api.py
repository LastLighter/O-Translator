"""FastAPI routes for O-Translator."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import ValidationError

from backend.artifacts import ARTIFACT_FILES, ArtifactStore
from backend.config import Settings
from backend.jobs import JobManager
from core_pipeline.errors import InvalidImageError, OTranslatorError, UploadTooLargeError
from core_pipeline.image_utils import decode_image
from core_pipeline.runtime import describe_runtime, resolve_providers
from core_pipeline.types import PipelineOptions

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

LANGUAGES = [
    {"code": "auto", "name": "Auto detect", "native": "自动检测"},
    {"code": "zh", "name": "Chinese", "native": "简体中文"},
    {"code": "en", "name": "English", "native": "English"},
    {"code": "ja", "name": "Japanese", "native": "日本語"},
    {"code": "ko", "name": "Korean", "native": "한국어"},
    {"code": "fr", "name": "French", "native": "Français"},
    {"code": "de", "name": "German", "native": "Deutsch"},
    {"code": "es", "name": "Spanish", "native": "Español"},
    {"code": "ru", "name": "Russian", "native": "Русский"},
]

TERMINAL = {"completed", "failed", "cancelled"}


def _effective_ocr(active: str, paddle: bool, onnx: bool, tesseract: bool) -> str:
    if active != "auto":
        return active
    if paddle:
        return "paddle"
    if onnx:
        return "onnx"
    if tesseract:
        return "tesseract"
    return "unavailable"


def _effective_inpaint(active: str, lama: bool, onnx: bool) -> str:
    if active != "auto":
        return active
    return "lama-onnx" if lama and onnx else "opencv-telea"


def _state(request: Request):
    return request.app.state


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _jobs(request: Request) -> JobManager:
    return request.app.state.jobs


def _store(request: Request) -> ArtifactStore:
    return request.app.state.store


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    state = _state(request)
    settings: Settings = state.settings
    pipeline = state.pipeline
    ocr_name = getattr(getattr(pipeline, "ocr", None), "name", "unavailable")
    inpaint_name = getattr(getattr(pipeline, "inpainter", None), "name", "unavailable")
    translator_name = getattr(getattr(pipeline, "translator", None), "name", "unavailable")
    lama_available = settings.lama_model_available
    onnx_available = bool(importlib.util.find_spec("onnxruntime"))
    paddle_available = bool(importlib.util.find_spec("paddleocr"))
    tesseract_wrapper = bool(importlib.util.find_spec("pytesseract"))
    onnx_ocr_available = bool(
        importlib.util.find_spec("rapidocr_onnxruntime")
        or importlib.util.find_spec("rapidocr")
    )
    runtime = describe_runtime()
    providers = resolve_providers(
        settings.lama_device,
        explicit=settings.lama_providers,
        available=runtime.get("providers_available", []),  # type: ignore[arg-type]
    )

    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.version,
        "engines": {
            "ocr": {
                "active": _effective_ocr(
                    ocr_name, paddle_available, onnx_ocr_available, tesseract_wrapper
                ),
                "paddle_installed": paddle_available,
                "tesseract_wrapper_installed": tesseract_wrapper,
                "onnx_installed": onnx_ocr_available,
                "configured": settings.ocr_engine,
                "providers": resolve_providers(
                    settings.ocr_device,
                    explicit=settings.ocr_providers,
                    available=runtime.get("providers_available", []),  # type: ignore[arg-type]
                ),
                "paddle_lang": settings.paddle_lang,
                "paddle_profile": settings.paddle_profile,
                "paddle_det_limit_side_len": settings.paddle_det_limit_side_len,
            },
            "inpainting": {
                "active": _effective_inpaint(inpaint_name, lama_available, onnx_available),
                "lama_weights_present": lama_available,
                "lama_path": str(settings.resolved_lama_path),
                "lama_source": settings.lama_model_source,
                "lama_precision": settings.lama_precision,
                "lama_device": settings.lama_device,
                "providers": providers,
                "fast_stddev": settings.inpaint_fast_stddev,
                "roi_padding": settings.inpaint_roi_padding,
                "progressive_preview": settings.progressive_preview,
                "onnxruntime_installed": onnx_available,
                "opencv_fallback": True,
                "download": (
                    downloader.state.snapshot()
                    if (downloader := getattr(state, "lama_download", None))
                    else None
                ),
            },
            "runtime": runtime,
            "translation": {
                "active": translator_name,
                "configured_provider": settings.effective_translation_provider,
                "deepseek_configured": settings.translation_configured,
                "model": settings.deepseek_model,
            },
        },
        "queue": {
            "workers": state.jobs.worker_count,
            "queued": state.jobs.queue_size,
            "active": state.jobs.active_jobs,
        },
        "limits": {
            "max_upload_mb": settings.max_upload_mb,
            "max_image_side": settings.max_image_side,
        },
        "startup_issues": state.startup_issues,
    }


@router.get("/languages")
async def languages() -> list[dict[str, str]]:
    return LANGUAGES


@router.get("/capabilities")
async def capabilities(request: Request) -> dict[str, object]:
    return await health(request)


@router.post("/jobs", status_code=202)
async def create_job(
    request: Request,
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("zh"),
    mode: str = Form("auto"),
    reading_order: str = Form("auto"),
    dilate_px: int | None = Form(None),
    font_scale: float = Form(1.0),
    translate: bool = Form(True),
) -> JSONResponse:
    settings = _settings(request)
    limit = settings.max_upload_mb * 1024 * 1024
    payload = await file.read(limit + 1)
    if len(payload) > limit:
        raise UploadTooLargeError(
            f"Image exceeds the {settings.max_upload_mb} MB upload limit.",
            details={"max_bytes": limit},
        )
    if not payload:
        raise InvalidImageError("The uploaded file is empty.")

    filename = (file.filename or "screenshot.png")[:180]
    # Validate and probe dimensions without retaining another full image.
    await asyncio.to_thread(
        decode_image,
        payload,
        max_side=settings.max_image_side,
        max_pixels=settings.max_image_pixels,
    )

    try:
        options = PipelineOptions(
            source_lang=source_lang,
            target_lang=target_lang,
            mode=mode,  # type: ignore[arg-type]
            reading_order=reading_order,  # type: ignore[arg-type]
            dilate_px=dilate_px,
            font_scale=font_scale,
            translate=translate,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    record = await _jobs(request).submit(
        payload, filename=filename, options=options
    )
    return JSONResponse(
        status_code=202,
        content=json.loads(record.model_dump_json()),
    )


@router.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str) -> dict[str, object]:
    record = _jobs(request).get(job_id)
    return json.loads(record.model_dump_json())


@router.get("/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str) -> StreamingResponse:
    manager = _jobs(request)
    manager.get(job_id)  # 404 before streaming starts

    async def stream():
        last_payload = ""
        heartbeat = 0
        while True:
            if await request.is_disconnected():
                break
            try:
                record = manager.get(job_id)
            except OTranslatorError:
                yield 'event: error\ndata: {"message":"job not found"}\n\n'
                break
            payload = record.model_dump_json()
            if payload != last_payload:
                last_payload = payload
                yield f"event: progress\ndata: {payload}\n\n"
                heartbeat = 0
            else:
                heartbeat += 1
                if heartbeat >= 20:
                    heartbeat = 0
                    yield ": keep-alive\n\n"
            if record.status in TERMINAL:
                yield f"event: done\ndata: {payload}\n\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/jobs/{job_id}/files/{filename}")
async def job_file(request: Request, job_id: str, filename: str):
    manager = _jobs(request)
    if not manager.has(job_id):
        raise HTTPException(status_code=404, detail="Job not found.")
    if filename not in ARTIFACT_FILES:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    path: Path = _store(request).file_path(job_id, filename)
    media_type = "image/png"
    if filename.endswith(".json"):
        media_type = "application/json"
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "no-store"})


@router.get("/jobs/{job_id}/download")
async def download_result(request: Request, job_id: str):
    manager = _jobs(request)
    if not manager.has(job_id):
        raise HTTPException(status_code=404, detail="Job not found.")
    path = _store(request).file_path(job_id, "result.png")
    return FileResponse(
        path,
        media_type="image/png",
        filename=f"o-translator-{job_id}.png",
    )
"""FastAPI application factory and server entry point."""

from __future__ import annotations

import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.api import router
from backend.artifacts import ArtifactStore
from backend.config import Settings, get_settings
from backend.jobs import JobManager
from core_pipeline.errors import OTranslatorError
from core_pipeline.inpainting import OpenCVInpainter, build_inpainter
from core_pipeline.model_store import configure_network, ensure_lama_downloader
from core_pipeline.ocr import DisabledOCRBackend, build_ocr_backend
from core_pipeline.pipeline import TranslationPipeline
from core_pipeline.runtime import activate_gpu_runtime, describe_runtime
from core_pipeline.translation import EchoTranslator, build_translator
from core_pipeline.typesetting import FontBook, Renderer

logger = logging.getLogger(__name__)

# Must happen before PaddleOCR (which may import onnxruntime for auxiliary
# features) or any provider probe imports the bundled CPU runtime.
activate_gpu_runtime()


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=False,
    )


def _build_pipeline(
    settings: Settings, issues: list[str]
) -> TranslationPipeline:
    try:
        ocr = build_ocr_backend(
            engine=settings.ocr_engine,
            paddle_lang=settings.paddle_lang,
            paddle_det_model=settings.paddle_det_model,
            paddle_rec_model=settings.paddle_rec_model,
            paddle_det_limit_side_len=settings.paddle_det_limit_side_len,
            paddle_profile=settings.paddle_profile,
            paddle_auto_fallback=settings.paddle_auto_fallback,
            min_confidence=settings.ocr_min_confidence,
            tesseract_lang=settings.tesseract_lang,
            tesseract_cmd=settings.tesseract_cmd,
            onnx_device=settings.ocr_device,
            onnx_providers=settings.ocr_providers,
        )
    except OTranslatorError as exc:
        ocr = DisabledOCRBackend()
        issues.append(f"OCR unavailable: {exc.user_message}")
        logger.warning("OCR unavailable at startup: %s", exc)

    try:
        inpainter = build_inpainter(
            engine=settings.inpaint_engine,
            lama_model_path=settings.resolved_lama_path,
            device=settings.lama_device,
            max_side=settings.inpaint_max_side,
            normalization=settings.lama_normalization,
            threads=settings.lama_threads,
            providers=settings.lama_providers,
            openvino_device=settings.lama_openvino_device,
            dml_device_id=settings.lama_dml_device_id,
            fast_stddev=settings.inpaint_fast_stddev,
            solid_stddev=settings.inpaint_solid_stddev,
            roi_padding=settings.inpaint_roi_padding,
            max_roi_patches=settings.inpaint_max_roi_patches,
            blend=settings.inpaint_blend,
        )
    except OTranslatorError as exc:
        inpainter = OpenCVInpainter()
        issues.append(f"LaMa unavailable: {exc.user_message}")
        logger.warning("LaMa unavailable at startup: %s", exc)

    provider = settings.effective_translation_provider
    try:
        translator = build_translator(
            provider=provider,
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            base_url=settings.deepseek_base_url,
            timeout_s=settings.translation_timeout_s,
            batch_size=settings.translation_batch_size,
            concurrency=settings.translation_concurrency,
            max_retries=settings.translation_max_retries,
            extra_prompt=settings.translation_extra_prompt,
        )
    except OTranslatorError as exc:
        translator = EchoTranslator()
        issues.append(f"Translation unavailable: {exc.user_message}")
        logger.warning("Translation unavailable at startup: %s", exc)

    fontbook = FontBook(settings.resolved_font_path)
    if fontbook.warning:
        issues.append(fontbook.warning)
    renderer = Renderer(
        fontbook,
        min_font_size=settings.render_min_font,
        font_scale=settings.render_font_scale,
    )
    logger.info("Pipeline configuration: %s", settings.describe())
    return TranslationPipeline(
        ocr=ocr,
        inpainter=inpainter,
        translator=translator,
        renderer=renderer,
        max_image_side=settings.max_image_side,
        max_image_pixels=settings.max_image_pixels,
        dilate_px=settings.dilate_px,
        translation_fallback_to_source=settings.translation_fallback_to_source,
        progressive_preview=settings.progressive_preview,
    )


def _warm_models(pipeline: TranslationPipeline, downloader) -> None:
    """Initialize OCR and LaMa off the request path.

    The first Paddle inference compiles oneDNN kernels and usually costs
    several seconds, which used to land on the user's first screenshot. OCR is
    warmed first because it dominates every job; LaMa follows when its weights
    are available.
    """
    warm_ocr = getattr(pipeline.ocr, "warm_up", None)
    if callable(warm_ocr):
        started = time.perf_counter()
        try:
            warm_ocr()
            logger.info(
                "OCR warm-up complete in %.1fs", time.perf_counter() - started
            )
        except Exception:  # pragma: no cover - hardware/runtime specific
            logger.warning("Background OCR warm-up failed", exc_info=True)
    if downloader is not None:
        while downloader.state.status == "downloading":
            time.sleep(1.0)
        if downloader.state.status != "ready":
            return
    time.sleep(1.0)
    warm_up = getattr(pipeline.inpainter, "warm_up", None)
    if callable(warm_up):
        try:
            warm_up()
        except Exception:  # pragma: no cover - hardware/runtime specific
            logger.warning("Background LaMa warm-up failed", exc_info=True)


def create_app(
    settings: Settings | None = None,
    *,
    pipeline_override: TranslationPipeline | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    startup_issues: list[str] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        settings.ensure_dirs()
        configure_network(settings)
        logger.info("ONNX Runtime: %s", describe_runtime())
        downloader = None
        if (
            pipeline_override is None
            and "pytest" not in sys.modules
            and settings.inpaint_engine in {"auto", "lama"}
            and settings.lama_auto_download
            and not settings.lama_model_available
        ):
            downloader = ensure_lama_downloader(settings)
            state = downloader.start()
            if state.status == "ready":
                logger.info("LaMa weights already present: %s", downloader.target)
            elif state.status == "downloading":
                logger.info(
                    "LaMa weights missing; downloading in the background to %s",
                    downloader.target,
                )
        elif settings.lama_model_available:
            logger.info(
                "LaMa weights ready (%s): %s",
                settings.lama_model_source,
                settings.resolved_lama_path,
            )
        if pipeline_override is not None:
            pipeline = pipeline_override
        else:
            pipeline = _build_pipeline(settings, startup_issues)
        store = ArtifactStore(settings.data_dir / "jobs")
        jobs = JobManager(
            pipeline=pipeline,
            store=store,
            workers=settings.queue_workers,
            capacity=settings.queue_capacity,
            history_limit=settings.job_history_limit,
        )
        await jobs.start()
        app.state.settings = settings
        app.state.pipeline = pipeline
        app.state.store = store
        app.state.jobs = jobs
        app.state.startup_issues = startup_issues
        app.state.lama_download = downloader
        if pipeline_override is None:
            threading.Thread(
                target=_warm_models,
                args=(pipeline, downloader),
                name="otr-lama-warmup",
                daemon=True,
            ).start()
        logger.info(
            "%s %s ready at http://%s:%d",
            settings.app_name,
            settings.version,
            settings.host,
            settings.port,
        )
        if startup_issues:
            for issue in startup_issues:
                logger.warning("Startup issue: %s", issue)
        try:
            yield
        finally:
            logger.info("Shutting down O-Translator")
            await jobs.stop()
            try:
                await pipeline.aclose()
            except Exception:
                logger.exception("Error while closing translator client")

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "Screenshot OCR, LaMa/OpenCV inpainting, DeepSeek batch translation, "
            "and adaptive typesetting."
        ),
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    origins = [item.strip() for item in settings.cors_origins.split(",") if item.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["http://127.0.0.1:8765", "http://localhost:8765"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.exception_handler(OTranslatorError)
    async def otranslator_error_handler(request: Request, exc: OTranslatorError):
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "code": "validation_error",
                "message": "Request validation failed.",
                "details": {"errors": jsonable_encoder(exc.errors())},
            },
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        return JSONResponse(
            status_code=404,
            content={"code": "artifact_not_found", "message": "Artifact not found."},
        )

    web_dir: Path = settings.client_web_dir
    if web_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def index():
            index_file = web_dir / "index.html"
            if not index_file.is_file():
                return JSONResponse(
                    status_code=500, content={"message": "Web UI files are missing."}
                )
            return FileResponse(index_file, media_type="text/html")

        @app.get("/favicon.ico", include_in_schema=False)
        async def favicon():
            icon = web_dir / "assets" / "favicon.svg"
            if icon.is_file():
                return FileResponse(icon, media_type="image/svg+xml")
            return RedirectResponse("/static/assets/favicon.svg")
    else:
        logger.warning("Web UI directory not found: %s", web_dir)

    return app


app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
"""In-process async job queue for the screenshot translation pipeline."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from backend.artifacts import ArtifactStore
from core_pipeline.errors import JobNotFoundError, OTranslatorError, QueueFullError
from core_pipeline.pipeline import TranslationPipeline
from core_pipeline.types import PipelineOptions

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobRecord(BaseModel):
    id: str
    status: JobStatus = "queued"
    stage: str = "queued"
    progress: int = 0
    message: str = "Queued"
    filename: str = ""
    options: PipelineOptions
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    warnings: list[str] = Field(default_factory=list)
    error: dict[str, object] | None = None
    result: dict[str, object] | None = None
    preview: dict[str, object] | None = None


class JobManager:
    def __init__(
        self,
        *,
        pipeline: TranslationPipeline,
        store: ArtifactStore,
        workers: int = 2,
        capacity: int = 32,
        history_limit: int = 100,
    ) -> None:
        self.pipeline = pipeline
        self.store = store
        self.worker_count = max(1, workers)
        self.capacity = max(1, capacity)
        self.history_limit = max(10, history_limit)
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self.capacity)
        self._records: dict[str, JobRecord] = {}
        self._payloads: dict[str, bytes] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._started = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._workers = [
            asyncio.create_task(self._worker(index), name=f"otranslator-worker-{index}")
            for index in range(self.worker_count)
        ]
        logger.info("Job queue started with %d workers", self.worker_count)

    async def stop(self) -> None:
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._started = False

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def active_jobs(self) -> int:
        return sum(1 for record in self._records.values() if record.status == "running")

    async def submit(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        options: PipelineOptions,
    ) -> JobRecord:
        async with self._lock:
            if self._queue.full():
                raise QueueFullError(
                    "The translation queue is full. Wait for a job to finish and retry."
                )
            job_id = self._new_job_id()
            while job_id in self._records:
                job_id = self._new_job_id()
            record = JobRecord(
                id=job_id,
                filename=filename,
                options=options,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
            self._records[job_id] = record
            self._payloads[job_id] = image_bytes
            try:
                self._queue.put_nowait(job_id)
            except asyncio.QueueFull as exc:
                self._records.pop(job_id, None)
                self._payloads.pop(job_id, None)
                raise QueueFullError(
                    "The translation queue is full. Wait for a job to finish and retry."
                ) from exc
            self._prune_history()
            return record.model_copy(deep=True)

    def get(self, job_id: str) -> JobRecord:
        record = self._records.get(job_id)
        if record is None:
            raise JobNotFoundError(f"Job {job_id} was not found or has expired.")
        return record.model_copy(deep=True)

    def has(self, job_id: str) -> bool:
        return job_id in self._records

    async def wait(self, job_id: str, timeout: float = 300.0) -> JobRecord:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.get(job_id)
            if record.status in {"completed", "failed", "cancelled"}:
                return record
            await asyncio.sleep(0.15)
        raise TimeoutError(f"Timed out waiting for job {job_id}")

    def _new_job_id(self) -> str:
        return f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"

    def _update(
        self,
        job_id: str,
        *,
        stage: str | None = None,
        progress: int | None = None,
        message: str | None = None,
        status: JobStatus | None = None,
        warnings: list[str] | None = None,
        error: dict[str, object] | None = None,
        result: dict[str, object] | None = None,
        preview: dict[str, object] | None = None,
    ) -> None:
        record = self._records.get(job_id)
        if record is None:
            return
        if stage is not None:
            record.stage = stage
        if progress is not None:
            record.progress = max(record.progress, min(100, progress))
        if message is not None:
            record.message = message
        if status is not None:
            record.status = status
        if warnings is not None:
            record.warnings = warnings
        if error is not None:
            record.error = error
        if result is not None:
            record.result = result
        if preview is not None:
            record.preview = preview
        record.updated_at = utcnow()

    def _progress_callback(self, job_id: str):
        def callback(stage: str, progress: int, message: str) -> None:
            self._update(
                job_id,
                stage=stage,
                progress=progress,
                message=message,
                status="running",
            )

        return callback

    async def _worker(self, index: int) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self._process(job_id)
            except asyncio.CancelledError:
                self._update(
                    job_id,
                    status="cancelled",
                    stage="cancelled",
                    message="Server is shutting down",
                )
                raise
            except Exception:  # defensive: _process handles its own errors
                logger.exception("Unhandled queue worker error for job %s", job_id)
            finally:
                self._queue.task_done()

    async def _process(self, job_id: str) -> None:
        record = self._records.get(job_id)
        payload = self._payloads.pop(job_id, None)
        if record is None or payload is None:
            return
        started = time.perf_counter()
        self._update(
            job_id,
            status="running",
            stage="decoding",
            progress=2,
            message="Starting pipeline",
        )
        try:
            async def publish_preview(
                image,
                blocks,
                regions,
                translations,
                timings,
            ) -> None:
                try:
                    payload = await asyncio.to_thread(
                        self.store.save_preview,
                        job_id,
                        image,
                        blocks,
                        regions,
                        translations,
                        timings,
                    )
                    self._update(
                        job_id,
                        stage="preview",
                        progress=82,
                        message="Translation preview ready",
                        preview=payload,
                    )
                except Exception:  # a preview write must not fail the job
                    logger.warning(
                        "Preview artifact could not be saved for job %s",
                        job_id,
                        exc_info=True,
                    )

            result = await self.pipeline.run(
                payload,
                record.options,
                job_id=job_id,
                progress=self._progress_callback(job_id),
                on_preview=publish_preview,
            )
            self._update(
                job_id,
                stage="saving",
                progress=97,
                message="Saving artifacts",
            )
            artifact_payload = await asyncio.to_thread(
                self.store.save_result, job_id, result
            )
            self._update(
                job_id,
                status="completed",
                stage="complete",
                progress=100,
                message="Complete",
                warnings=result.warnings,
                result=artifact_payload,
            )
            logger.info(
                "Job %s completed in %.2fs", job_id, time.perf_counter() - started
            )
        except asyncio.CancelledError:
            self._update(
                job_id,
                status="cancelled",
                stage="cancelled",
                message="Cancelled",
            )
            raise
        except OTranslatorError as exc:
            logger.warning("Job %s failed: %s", job_id, exc)
            self._update(
                job_id,
                status="failed",
                stage="failed",
                progress=100,
                message=exc.user_message,
                error=exc.to_payload(),
            )
        except Exception as exc:
            logger.exception("Job %s failed unexpectedly", job_id)
            self._update(
                job_id,
                status="failed",
                stage="failed",
                progress=100,
                message="Internal pipeline error.",
                error={
                    "code": "internal_error",
                    "message": "Internal pipeline error. Check the server log.",
                    "details": {"error": str(exc)},
                },
            )
        finally:
            self._payloads.pop(job_id, None)

    def _prune_history(self) -> None:
        if len(self._records) <= self.history_limit:
            return
        terminal = [
            record
            for record in self._records.values()
            if record.status in {"completed", "failed", "cancelled"}
        ]
        terminal.sort(key=lambda record: record.updated_at)
        for record in terminal[: max(0, len(self._records) - self.history_limit)]:
            self._records.pop(record.id, None)
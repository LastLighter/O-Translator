"""Job artifact persistence and API result shaping."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from PIL import Image

from core_pipeline.image_utils import save_png_atomic
from core_pipeline.types import PipelineResult

ARTIFACT_FILES = {
    "original.png",
    "clean.png",
    "mask.png",
    "result.png",
    "preview.png",
    "metadata.json",
}


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        # Job ids are generated server-side; still guard against traversal.
        safe = "".join(ch for ch in job_id if ch.isalnum() or ch in "-_")
        if not safe or safe != job_id:
            raise ValueError("Invalid job id")
        path = self.root / safe
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_result(self, job_id: str, result: PipelineResult) -> dict[str, object]:
        directory = self.job_dir(job_id)
        save_png_atomic(result.original, directory / "original.png")
        save_png_atomic(result.clean, directory / "clean.png")
        save_png_atomic(result.mask, directory / "mask.png")
        save_png_atomic(result.rendered, directory / "result.png")

        blocks = [block.model_dump(mode="json") for block in result.blocks]
        regions = self._regions_payload(result.regions, result.translations)

        payload: dict[str, object] = {
            "job_id": job_id,
            "image_url": self.url(job_id, "result.png"),
            "original_url": self.url(job_id, "original.png"),
            "clean_url": self.url(job_id, "clean.png"),
            "mask_url": self.url(job_id, "mask.png"),
            "width": result.processed_size[0],
            "height": result.processed_size[1],
            "source_width": result.source_size[0],
            "source_height": result.source_size[1],
            "empty": result.empty,
            "blocks": blocks,
            "regions": regions,
            "engines": {
                "ocr": result.ocr_engine,
                "inpainting": result.inpaint_engine,
                "inpaint_strategy": result.inpaint_strategy,
            },
            "translation": {
                "provider": result.translation_provider,
                "model": result.translation_model,
                "degraded": result.degraded,
                "usage": result.translation_usage,
            },
            "options": result.options.model_dump(mode="json"),
            "timings_ms": result.timings_ms,
            "warnings": result.warnings,
        }
        self._write_json(directory / "metadata.json", payload)
        return payload

    def save_preview(
        self,
        job_id: str,
        image: Image.Image,
        blocks: list[object],
        regions: list[object],
        translations: dict[int, str],
        timings_ms: dict[str, float] | None = None,
    ) -> dict[str, object]:
        """Persist the pass-1 render while LaMa finishes in the background.

        The preview is a normal artifact (``preview.png``) so the client can
        fetch it through the same authenticated-by-locality file route.
        """
        directory = self.job_dir(job_id)
        save_png_atomic(image, directory / "preview.png")
        return {
            "image_url": self.url(job_id, "preview.png"),
            "width": image.width,
            "height": image.height,
            "preview": True,
            "blocks": [
                block.model_dump(mode="json")
                if hasattr(block, "model_dump")
                else block
                for block in blocks
            ],
            "regions": self._regions_payload(regions, translations),
            "timings_ms": timings_ms or {},
        }

    @staticmethod
    def _regions_payload(
        regions: list[object], translations: dict[int, str]
    ) -> list[dict[str, object]]:
        payload: list[dict[str, object]] = []
        for region in regions:
            item = region.model_dump(mode="json")  # type: ignore[attr-defined]
            box = region.bbox  # type: ignore[attr-defined]
            item["bbox"] = [box.x, box.y, box.w, box.h]
            item["translation"] = translations.get(region.id, region.text)  # type: ignore[attr-defined]
            payload.append(item)
        return payload

    def write_metadata(self, job_id: str, payload: dict[str, object]) -> None:
        self._write_json(self.job_dir(job_id) / "metadata.json", payload)

    @staticmethod
    def url(job_id: str, filename: str) -> str:
        return f"/api/jobs/{job_id}/files/{filename}"

    def file_path(self, job_id: str, filename: str) -> Path:
        if filename not in ARTIFACT_FILES:
            raise FileNotFoundError(filename)
        path = self.job_dir(job_id) / filename
        if not path.is_file():
            raise FileNotFoundError(filename)
        return path

    def exists(self, job_id: str, filename: str) -> bool:
        try:
            return self.file_path(job_id, filename).is_file()
        except (FileNotFoundError, ValueError):
            return False

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.stem}-", suffix=".json", dir=str(path.parent)
        )
        os.close(fd)
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
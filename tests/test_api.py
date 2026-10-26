from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app
from core_pipeline.inpainting import OpenCVInpainter
from core_pipeline.ocr import OCRBackend
from core_pipeline.pipeline import TranslationPipeline
from core_pipeline.translation import EchoTranslator
from core_pipeline.types import OcrOutput, TextRegion
from core_pipeline.typesetting import FontBook, Renderer


class FakeOCR(OCRBackend):
    name = "fake-ocr"

    def __init__(self, regions):
        self.regions = regions

    def read(self, image):
        return OcrOutput(regions=self.regions, engine=self.name, elapsed_ms=1.0)


def _regions() -> list[TextRegion]:
    return [
        TextRegion(
            id=0,
            text="测试文本",
            confidence=0.99,
            polygon=[(80, 70), (280, 70), (280, 120), (80, 120)],
        )
    ]


@pytest.fixture
def app(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        client_web_dir=tmp_path / "web",
        translation_provider="echo",
        queue_workers=1,
    )
    settings.ensure_dirs()
    pipeline = TranslationPipeline(
        ocr=FakeOCR(_regions()),
        inpainter=OpenCVInpainter(),
        translator=EchoTranslator(),
        renderer=Renderer(FontBook()),
    )
    return create_app(settings, pipeline_override=pipeline)


def _wait(client: TestClient, job_id: str, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_health_and_languages(app):
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        languages = client.get("/api/languages").json()
        assert any(item["code"] == "ja" for item in languages)


def test_create_translate_and_fetch_artifacts(app, sample_png):
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            files={"file": ("shot.png", sample_png, "image/png")},
            data={
                "source_lang": "zh",
                "target_lang": "en",
                "mode": "manga",
                "reading_order": "rtl",
                "dilate_px": "4",
                "font_scale": "1.0",
                "translate": "true",
            },
        )
        assert response.status_code == 202, response.text
        job = response.json()
        assert job["status"] in {"queued", "running"}
        final = _wait(client, job["id"])
        assert final["status"] == "completed", final
        result = final["result"]
        assert result["image_url"].endswith("result.png")
        assert result["engines"]["ocr"] == "fake-ocr"
        assert result["translation"]["provider"] == "echo"
        assert len(result["regions"]) == 1
        assert final["preview"]["image_url"].endswith("preview.png")
        preview = client.get(final["preview"]["image_url"])
        assert preview.status_code == 200
        assert preview.headers["content-type"] == "image/png"

        image = client.get(result["image_url"])
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/png"
        mask = client.get(result["mask_url"])
        assert mask.status_code == 200
        download = client.get(f"/api/jobs/{job['id']}/download")
        assert download.status_code == 200
        assert "attachment" in download.headers["content-disposition"]

        events = client.get(f"/api/jobs/{job['id']}/events")
        assert events.status_code == 200
        assert "event: done" in events.text


def test_invalid_image_returns_structured_error(app):
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            files={"file": ("bad.png", b"not really a png", "image/png")},
        )
        assert response.status_code == 400
        payload = response.json()
        assert payload["code"] == "invalid_image"


def test_empty_upload_returns_structured_error(app):
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_image"


def test_missing_upload_returns_validation_envelope(app):
    with TestClient(app) as client:
        response = client.post("/api/jobs")
        assert response.status_code == 422
        payload = response.json()
        assert payload["code"] == "validation_error"
        assert payload["details"]["errors"]


def test_unknown_job_returns_404(app):
    with TestClient(app) as client:
        response = client.get("/api/jobs/does-not-exist")
        assert response.status_code == 404
        assert response.json()["code"] == "job_not_found"


def test_upload_limit(app, sample_png, tmp_path):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data2",
        client_web_dir=tmp_path / "web",
        translation_provider="echo",
        max_upload_mb=0,
    )
    settings.ensure_dirs()
    pipeline = TranslationPipeline(
        ocr=FakeOCR(_regions()),
        inpainter=OpenCVInpainter(),
        translator=EchoTranslator(),
        renderer=Renderer(FontBook()),
    )
    limited = create_app(settings, pipeline_override=pipeline)
    with TestClient(limited) as client:
        response = client.post(
            "/api/jobs",
            files={"file": ("shot.png", sample_png, "image/png")},
        )
        assert response.status_code == 413
        assert response.json()["code"] == "upload_too_large"
from __future__ import annotations

from core_pipeline.model_store import (
    DownloadState,
    ModelDownloader,
    lama_urls,
)


class _Settings:
    def __init__(self, target, *, endpoint=None, url=None):
        self.resolved_lama_path = target
        self.hf_endpoint = endpoint
        self.lama_download_url = url


def test_downloader_accepts_existing_model(tmp_path):
    target = tmp_path / "lama_fp32.onnx"
    target.write_bytes(b"x" * 32)
    downloader = ModelDownloader(target, urls=[], min_bytes=16)
    state = downloader.start()
    assert state.status == "ready"
    assert downloader.state.downloaded == 32
    # A second start must not spawn a downloading thread.
    assert downloader.start().status == "ready"


def test_lama_urls_honours_endpoint_and_override(tmp_path):
    urls = lama_urls(_Settings(tmp_path / "lama.onnx", endpoint="https://hf-mirror.com/"))
    assert urls[0] == (
        "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
    )
    override = lama_urls(
        _Settings(tmp_path / "lama.onnx", url="https://example.com/lama.onnx")
    )
    assert override == ["https://example.com/lama.onnx"]


def test_download_state_snapshot_progress():
    state = DownloadState(
        status="downloading",
        downloaded=5 * 1024 * 1024,
        total=10 * 1024 * 1024,
    )
    snapshot = state.snapshot()
    assert snapshot["percent"] == 50.0
    assert snapshot["downloaded_human"] == "5.0 MB"
    assert snapshot["status"] == "downloading"

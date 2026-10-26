"""Optional model downloads (currently LaMa ONNX) with progress tracking.

The backend starts a daemon downloader when the LaMa weights are missing so the
first screenshot does not silently fall back to OpenCV forever. Downloads honour
``OTR_PROXY``/``HTTP(S)_PROXY`` and an optional Hugging Face mirror
(``OTR_HF_ENDPOINT``), and fall back across mirrors. The ONNX graph is only
reported as ready after the file is complete and large enough to be a real
model rather than an HTML error page.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MIN_LAMA_BYTES = 20 * 1024 * 1024
HF_LAMA_REPO_PATH = "Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
DEFAULT_LAMA_URLS = (
    "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
)


def _human(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def configure_network(settings: Any) -> None:
    """Apply proxy/mirror settings before RapidOCR or downloads run.

    RapidOCR's ModelScope downloads (via ``requests``) read ``HTTP(S)_PROXY``;
    the LaMa downloader also honours ``HF_ENDPOINT``. Clash/V2Ray users who only
    configured a browser extension can set ``OTR_PROXY=...:7897`` and have both
    the LaMa downloader and RapidOCR model downloads use it.
    """
    proxy = getattr(settings, "resolved_proxy", None)
    if proxy:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            os.environ[name] = proxy
        # Keep local API/health checks direct.
        no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
        local = {"127.0.0.1", "localhost", "::1"}
        existing = {item.strip() for item in no_proxy.split(",") if item.strip()}
        if not local.issubset(existing):
            os.environ["NO_PROXY"] = ",".join(sorted(existing | local))
        logger.info("Using explicit network proxy %s for model downloads", proxy)
    endpoint = getattr(settings, "hf_endpoint", None)
    if endpoint:
        normalized = str(endpoint).strip().rstrip("/")
        if normalized:
            os.environ.setdefault("HF_ENDPOINT", normalized)
            logger.info("Using Hugging Face endpoint %s", normalized)


def _proxy_map(explicit_proxy: str | None) -> dict[str, str]:
    if explicit_proxy:
        proxy = explicit_proxy.strip()
        return {"http": proxy, "https": proxy}
    try:
        return dict(urllib.request.getproxies())
    except Exception:  # pragma: no cover - platform specific
        return {}


def download_file(
    url: str,
    target: Path,
    *,
    proxy: str | None = None,
    timeout: float = 60.0,
    min_bytes: int = MIN_LAMA_BYTES,
    progress: Any = None,
) -> int:
    """Download ``url`` to ``target`` atomically and return the byte count."""
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "O-Translator/0.2 (+https://github.com/)",
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
        },
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(_proxy_map(proxy))
    )
    logger.info("Downloading %s", url)
    fd, temporary = tempfile.mkstemp(
        prefix="lama-", suffix=".part", dir=str(target.parent)
    )
    downloaded = 0
    last_report = 0.0
    try:
        with (
            os.fdopen(fd, "wb") as handle,
            opener.open(request, timeout=timeout) as response,
        ):
            total = int(response.headers.get("Content-Length") or 0)
            if progress:
                progress(0, total, url)
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if progress and now - last_report > 0.35:
                    last_report = now
                    progress(downloaded, total, url)
        if downloaded < min_bytes:
            raise RuntimeError(
                f"Downloaded file is only {_human(downloaded)}; expected a LaMa "
                "ONNX model of roughly 200 MB. The mirror may have returned an "
                "HTML error page."
            )
        os.replace(temporary, target)
        if progress:
            progress(downloaded, downloaded, url)
        logger.info("Saved %s (%s)", target, _human(downloaded))
        return downloaded
    finally:
        if os.path.exists(temporary):
            with contextlib.suppress(OSError):
                os.unlink(temporary)


@dataclass
class DownloadState:
    status: str = "idle"  # idle | downloading | ready | failed
    path: str = ""
    downloaded: int = 0
    total: int = 0
    source: str | None = None
    error: str | None = None
    started_at: float = 0.0
    updated_at: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def snapshot(self) -> dict[str, object]:
        percent = (
            round(min(100.0, self.downloaded / self.total * 100.0), 1)
            if self.total
            else None
        )
        return {
            "status": self.status,
            "path": self.path,
            "downloaded": self.downloaded,
            "total": self.total,
            "percent": percent,
            "downloaded_human": _human(self.downloaded),
            "total_human": _human(self.total),
            "source": self.source,
            "error": self.error,
            "warnings": list(self.warnings),
        }


class ModelDownloader:
    """One background downloader per target path."""

    def __init__(
        self,
        target: Path,
        *,
        urls: list[str] | tuple[str, ...],
        proxy: str | None = None,
        min_bytes: int = MIN_LAMA_BYTES,
    ) -> None:
        self.target = Path(target)
        self.urls = [url for url in urls if url]
        self.proxy = proxy
        self.min_bytes = min_bytes
        self.state = DownloadState(path=str(self.target))
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def already_present(self) -> bool:
        try:
            return (
                self.target.is_file()
                and self.target.stat().st_size >= self.min_bytes
            )
        except OSError:
            return False

    def start(self) -> DownloadState:
        with self._lock:
            if self.already_present:
                self.state.status = "ready"
                self.state.path = str(self.target)
                self.state.total = self.state.downloaded = self.target.stat().st_size
                self.state.updated_at = time.time()
                return self.state
            if self._thread is not None and self._thread.is_alive():
                return self.state
            if not self.urls:
                self.state.status = "failed"
                self.state.error = "No download source is configured."
                return self.state
            self.state.status = "downloading"
            self.state.started_at = time.time()
            self.state.updated_at = self.state.started_at
            self.state.error = None
            self._thread = threading.Thread(
                target=self._run,
                name="otr-lama-download",
                daemon=True,
            )
            self._thread.start()
            return self.state

    def _progress(self, downloaded: int, total: int, url: str) -> None:
        with self._lock:
            self.state.downloaded = downloaded
            self.state.total = total
            self.state.source = url
            self.state.updated_at = time.time()

    def _run(self) -> None:
        last_error: str | None = None
        for url in self.urls:
            try:
                download_file(
                    url,
                    self.target,
                    proxy=self.proxy,
                    min_bytes=self.min_bytes,
                    progress=self._progress,
                )
                if self.already_present:
                    with self._lock:
                        self.state.status = "ready"
                        self.state.source = url
                        self.state.downloaded = self.state.total = (
                            self.target.stat().st_size
                        )
                        self.state.updated_at = time.time()
                    return
                last_error = "Downloaded file did not pass the size check."
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                OSError,
                RuntimeError,
            ) as exc:
                last_error = f"{url}: {exc}"
                logger.warning("LaMa download failed from %s: %s", url, exc)
        with self._lock:
            self.state.status = "failed"
            self.state.error = last_error or "Unknown download error."
            self.state.updated_at = time.time()
        logger.error("LaMa auto-download failed: %s", last_error)


_DOWNLOADERS: dict[str, ModelDownloader] = {}
_DOWNLOADERS_LOCK = threading.Lock()


def lama_urls(settings: Any) -> list[str]:
    explicit = getattr(settings, "lama_download_url", None)
    if explicit:
        return [str(explicit)]
    urls: list[str] = []
    endpoint = getattr(settings, "hf_endpoint", None)
    if endpoint:
        base = str(endpoint).strip().rstrip("/")
        if base:
            urls.append(f"{base}/{HF_LAMA_REPO_PATH}")
    for url in DEFAULT_LAMA_URLS:
        if url not in urls:
            urls.append(url)
    return urls


def ensure_lama_downloader(settings: Any) -> ModelDownloader:
    """Return the singleton downloader for the configured LaMa path.

    The downloader writes to the writable model path even when a read-only
    optimized graph is bundled inside a packaged build, so an explicit
    ``OTR_LAMA_AUTO_DOWNLOAD=true`` can still upgrade a partial installation.
    """
    target = getattr(settings, "lama_download_path", None)
    if target is None:
        target = settings.resolved_lama_path
    target = Path(target)
    key = str(target).lower()
    with _DOWNLOADERS_LOCK:
        downloader = _DOWNLOADERS.get(key)
        if downloader is None:
            downloader = ModelDownloader(
                target,
                urls=lama_urls(settings),
                proxy=getattr(settings, "resolved_proxy", None),
            )
            _DOWNLOADERS[key] = downloader
        return downloader
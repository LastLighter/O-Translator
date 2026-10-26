"""HTTP API access, backend process management, and QRunnable helpers."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from client.config import API_URL, AUTOSTART, BACKEND_LOG, PROJECT_ROOT, mime_for


class ApiError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, base_url: str = API_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def health(self, timeout: float = 3.0) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(f"{self.base_url}/api/health")
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            raise ApiError(f"无法连接本地服务：{exc}") from exc

    def create_job(
        self,
        file_path: Path,
        *,
        source_lang: str,
        target_lang: str,
        mode: str,
        reading_order: str,
        dilate_px: int,
        font_scale: float,
        translate: bool,
    ) -> dict[str, Any]:
        content = file_path.read_bytes()
        return self.create_job_bytes(
            content,
            filename=file_path.name,
            mime=mime_for(file_path),
            source_lang=source_lang,
            target_lang=target_lang,
            mode=mode,
            reading_order=reading_order,
            dilate_px=dilate_px,
            font_scale=font_scale,
            translate=translate,
        )

    def create_job_bytes(
        self,
        content: bytes,
        *,
        filename: str,
        mime: str = "image/png",
        source_lang: str,
        target_lang: str,
        mode: str,
        reading_order: str,
        dilate_px: int,
        font_scale: float,
        translate: bool,
    ) -> dict[str, Any]:
        data = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "mode": mode,
            "reading_order": reading_order,
            "dilate_px": str(dilate_px),
            "font_scale": f"{font_scale:.2f}",
            "translate": "true" if translate else "false",
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.base_url}/api/jobs",
                    files={"file": (filename, content, mime)},
                    data=data,
                )
                payload = _json_or_message(response)
                if not response.is_success:
                    raise ApiError(
                        payload.get("message")
                        or payload.get("detail")
                        or "任务创建失败"
                    )
                return payload
        except httpx.HTTPError as exc:
            raise ApiError(f"上传失败：{exc}") from exc

    def get_job(self, job_id: str) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=20.0) as client:
                response = client.get(f"{self.base_url}/api/jobs/{job_id}")
                payload = _json_or_message(response)
                if not response.is_success:
                    raise ApiError(payload.get("message") or f"HTTP {response.status_code}")
                return payload
        except httpx.HTTPError as exc:
            raise ApiError(f"获取任务状态失败：{exc}") from exc

    def fetch_image(self, relative_url: str) -> bytes:
        try:
            with httpx.Client(timeout=45.0) as client:
                response = client.get(f"{self.base_url}{relative_url}")
                response.raise_for_status()
                return response.content
        except httpx.HTTPError as exc:
            raise ApiError(f"下载结果失败：{exc}") from exc


def _json_or_message(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {"message": response.text or f"HTTP {response.status_code}"}
    if isinstance(payload, dict):
        return payload
    return {"message": str(payload)}


class BackendManager:
    """Reuse a running server or spawn one with the current Python interpreter."""

    def __init__(self, api: ApiClient) -> None:
        self.api = api
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle = None
        self._lock = threading.Lock()
        self._closed = False

    def ensure(self, timeout: float = 60.0) -> dict[str, Any]:
        if self._closed:
            raise ApiError("客户端正在退出。")
        try:
            return self.api.health(timeout=2.0)
        except ApiError:
            pass
        if not AUTOSTART:
            raise ApiError(
                f"后端未运行（{self.api.base_url}）。请先启动 python -m backend.main。"
            )
        with self._lock:
            if self._closed:
                raise ApiError("客户端正在退出。")
            if self.process is None or self.process.poll() is not None:
                self._spawn()
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self._closed:
                raise ApiError("客户端正在退出。")
            if self.process is not None and self.process.poll() is not None:
                raise ApiError("后端进程启动失败。请查看 data/desktop-backend.log。")
            try:
                return self.api.health(timeout=2.0)
            except ApiError as exc:
                last_error = exc
                time.sleep(0.5)
        raise ApiError(f"等待后端启动超时：{last_error}")

    def _spawn(self) -> None:
        BACKEND_LOG.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = open(BACKEND_LOG, "ab", buffering=0)
        env = os.environ.copy()
        parsed = urlparse(self.api.base_url)
        if parsed.hostname:
            env["OTR_HOST"] = parsed.hostname
        if parsed.port:
            env["OTR_PORT"] = str(parsed.port)
        flags = 0
        if sys.platform == "win32":
            flags = 0x08000000  # CREATE_NO_WINDOW
        if getattr(sys, "frozen", False):
            # Packaged build: relaunch the exe itself in backend-only mode.
            command = [sys.executable, "--backend"]
        else:
            command = [sys.executable, "-m", "backend.main"]
        self.process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=self.log_handle,
            stderr=self.log_handle,
            creationflags=flags,
        )

    def close(self) -> None:
        self._closed = True
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.log_handle is not None:
            with suppress(OSError):
                self.log_handle.close()


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    done = Signal()


class Worker(QRunnable):
    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        result = None
        error: str | None = None
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
        try:
            if error is not None:
                self.signals.error.emit(error)
            else:
                self.signals.result.emit(result)
        except RuntimeError:
            pass
        finally:
            with suppress(RuntimeError):
                self.signals.done.emit()
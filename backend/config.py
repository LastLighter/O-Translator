"""Application configuration for O-Translator.

All values can be provided through environment variables or a local ``.env``
file. ``OTR_*`` is the canonical prefix; ``DEEPSEEK_API_KEY`` is accepted as a
convenience alias so existing shells keep working.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _is_frozen() -> bool:
    """True when running from a PyInstaller build."""
    return bool(getattr(sys, "frozen", False))


def _app_root() -> Path:
    """Writable root: the exe folder when frozen, the repo root otherwise."""
    if _is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _bundle_root() -> Path:
    """Read-only bundled resources root inside a PyInstaller build."""
    if _is_frozen():
        return Path(getattr(sys, "_MEIPASS", _app_root()))
    return _app_root()


BASE_DIR = _app_root()
BUNDLE_DIR = _bundle_root()

DEFAULT_LAMA_MODEL_PATH = BASE_DIR / "models" / "lama" / "lama_fp32.onnx"
MIN_LAMA_BYTES = 20 * 1024 * 1024


def _load_persisted_user_env(name: str) -> None:
    """Fallback for processes launched from a stale Explorer environment.

    ``setx``/``[Environment]::SetEnvironmentVariable`` write the user variable to
    the registry, but already-running processes (including Explorer) do not see
    it until they restart. Reading the persisted value keeps both packaged and
    source-run clients working immediately after the variable is set. The normal
    process environment always wins.
    """
    if os.environ.get(name):
        return
    if sys.platform != "win32":
        return
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return
    if isinstance(value, str) and value.strip():
        os.environ[name] = value.strip()


_load_persisted_user_env("DEEPSEEK_API_KEY")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OTR_",
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # --- application -----------------------------------------------------
    app_name: str = "O-Translator"
    version: str = "0.1.0"
    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "INFO"
    data_dir: Path = BASE_DIR / "data"
    models_dir: Path = BASE_DIR / "models"
    client_web_dir: Path = BUNDLE_DIR / "client" / "web"
    cors_origins: str = "http://127.0.0.1:8765,http://localhost:8765"

    # --- ingestion limits ------------------------------------------------
    max_upload_mb: int = 20
    max_image_side: int = 2400
    max_image_pixels: int = 40_000_000
    queue_workers: int = 2
    queue_capacity: int = 32
    job_history_limit: int = 100

    # --- OCR -------------------------------------------------------------
    ocr_engine: Literal["auto", "paddle", "onnx", "tesseract", "none"] = "auto"
    ocr_device: Literal["auto", "cpu", "cuda", "openvino", "dml", "directml"] = "auto"
    ocr_providers: str | None = None
    paddle_lang: str = "ch"
    paddle_det_model: str | None = None
    paddle_rec_model: str | None = None
    paddle_det_limit_side_len: int = 1280
    paddle_profile: Literal["auto", "fast", "quality"] = "auto"
    paddle_auto_fallback: bool = True
    ocr_min_confidence: float = 0.5
    tesseract_lang: str = "chi_sim+eng"
    tesseract_cmd: str | None = None

    # --- inpainting ------------------------------------------------------
    inpaint_engine: Literal["auto", "lama", "opencv"] = "auto"
    lama_model_path: Path = DEFAULT_LAMA_MODEL_PATH
    lama_precision: Literal["auto", "fp32", "fp16", "int8"] = "auto"
    lama_device: Literal["auto", "cpu", "cuda", "openvino", "dml", "directml"] = "auto"
    lama_providers: str | None = None
    lama_openvino_device: str = "CPU"
    lama_dml_device_id: int = 0
    lama_normalization: Literal["zero_one", "minus_one_one", "imagenet"] = "zero_one"
    inpaint_max_side: int = 1024
    # 0 = auto (capped at 4): ORT's FFT/Einsum-heavy LaMa graph stops scaling
    # after a few threads and starves OCR when it grabs every core.
    lama_threads: int = 0
    dilate_px: int = 7
    lama_auto_download: bool = True
    lama_download_url: str | None = None

    # --- adaptive inpainting --------------------------------------------
    # Regions whose surrounding-background stddev is below this threshold use
    # a local Telea/median fill instead of LaMa. ``solid`` is the stricter
    # threshold for a plain median-colour fill (no Telea at all).
    inpaint_fast_stddev: float = 24.0
    inpaint_solid_stddev: float = 3.0
    inpaint_roi_padding: int = 16
    inpaint_max_roi_patches: int = 1
    inpaint_blend: Literal["alpha", "seamless"] = "alpha"
    # Pass 1 (text over the fast background) is emitted while LaMa finishes.
    progressive_preview: bool = True

    # --- network ---------------------------------------------------------
    # PaddleOCR model downloads and the LaMa downloader honour these. Leave
    # them empty to use the system proxy (Clash/V2Ray system mode) or a direct
    # connection. Example: OTR_PROXY=http://127.0.0.1:7897
    http_proxy: str | None = None
    https_proxy: str | None = None
    proxy: str | None = None
    hf_endpoint: str | None = None

    # --- translation -----------------------------------------------------
    translation_provider: Literal["auto", "deepseek", "echo"] = "auto"
    deepseek_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OTR_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"),
    )
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    translation_timeout_s: float = 60.0
    translation_batch_size: int = 40
    translation_concurrency: int = 3
    translation_max_retries: int = 3
    translation_fallback_to_source: bool = True
    translation_extra_prompt: str = ""

    # --- rendering -------------------------------------------------------
    font_path: str | None = None
    render_min_font: int = 7
    render_font_scale: float = 1.0

    # --- desktop client --------------------------------------------------
    api_url: str = "http://127.0.0.1:8765"
    hotkey: str = "f2"
    client_autostart_backend: bool = True

    # ------------------------------------------------------------------
    @property
    def effective_translation_provider(self) -> str:
        if self.translation_provider == "deepseek":
            return "deepseek"
        if self.translation_provider == "echo":
            return "echo"
        return "deepseek" if self.deepseek_api_key else "echo"

    @property
    def translation_configured(self) -> bool:
        return self.effective_translation_provider == "deepseek" and bool(
            self.deepseek_api_key
        )

    @property
    def requested_lama_path(self) -> Path:
        """Absolute path of the configured model (download target)."""
        path = Path(self.lama_model_path)
        return path if path.is_absolute() else BASE_DIR / path

    def _lama_candidate_names(self) -> list[str]:
        path = self.requested_lama_path
        stem, suffix = path.stem, path.suffix
        for marker in ("_int8", "_fp16", "_sim", "_optimized"):
            if stem.endswith(marker):
                stem = stem[: -len(marker)]
                break
        by_precision = {
            "fp32": [f"{stem}_sim{suffix}", f"{stem}_optimized{suffix}"],
            "fp16": [f"{stem}_fp16{suffix}", f"{stem}_sim{suffix}"],
            "int8": [f"{stem}_int8{suffix}", f"{stem}_sim{suffix}"],
            "auto": [
                f"{stem}_sim{suffix}",
                f"{stem}_fp16{suffix}",
                f"{stem}_int8{suffix}",
                f"{stem}_optimized{suffix}",
            ],
        }
        names = by_precision.get(self.lama_precision, by_precision["auto"])
        names.append(path.name)
        return list(dict.fromkeys(names))

    def _lama_search_roots(self) -> list[Path]:
        requested = self.requested_lama_path
        roots = [requested.parent]
        # Packaged builds keep read-only data in ``_internal`` (``_MEIPASS``)
        # while the writable models dir sits next to the exe. Check both so a
        # bundled, pre-optimized LaMa graph is used without any download.
        if requested == DEFAULT_LAMA_MODEL_PATH and BUNDLE_DIR != BASE_DIR:
            bundled = BUNDLE_DIR / "models" / "lama"
            if bundled not in roots:
                roots.append(bundled)
        return roots

    @property
    def lama_model_candidates(self) -> list[Path]:
        paths: list[Path] = []
        for root in self._lama_search_roots():
            for name in self._lama_candidate_names():
                candidate = root / name
                if candidate not in paths:
                    paths.append(candidate)
        return paths

    @property
    def resolved_lama_path(self) -> Path:
        """Best available LaMa graph, including bundled/pre-optimized siblings.

        An explicitly configured model path is honored when it exists; the
        default path expands to ``_sim``/``_fp16``/``_int8`` siblings so a
        packaged build can ship only the optimized graph.
        """
        requested = self.requested_lama_path
        explicit = requested != DEFAULT_LAMA_MODEL_PATH
        if explicit and requested.is_file():
            return requested
        for candidate in self.lama_model_candidates:
            try:
                if (
                    candidate.is_file()
                    and candidate.stat().st_size >= MIN_LAMA_BYTES
                ):
                    return candidate
            except OSError:
                continue
        return requested

    @property
    def lama_download_path(self) -> Path:
        """Writable path used by the background downloader."""
        return self.requested_lama_path

    @property
    def lama_model_available(self) -> bool:
        resolved = self.resolved_lama_path
        try:
            return (
                resolved.is_file()
                and resolved.stat().st_size >= MIN_LAMA_BYTES
            )
        except OSError:
            return False

    @property
    def lama_model_source(self) -> str:
        resolved = self.resolved_lama_path.resolve()
        try:
            resolved.relative_to(BUNDLE_DIR.resolve())
            if BUNDLE_DIR.resolve() != BASE_DIR.resolve():
                return "bundled"
        except (OSError, ValueError):
            pass
        return "local" if resolved.is_file() else "missing"

    @property
    def resolved_proxy(self) -> str | None:
        """Single proxy URL used by model downloads and PaddleX."""
        for candidate in (self.https_proxy, self.http_proxy, self.proxy):
            if candidate and candidate.strip():
                return candidate.strip()
        return None

    @property
    def resolved_font_path(self) -> str | None:
        if not self.font_path:
            return None
        path = Path(self.font_path)
        if not path.is_absolute():
            path = BASE_DIR / path
        return str(path) if path.exists() else None

    def ensure_dirs(self) -> None:
        """Create writable runtime directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "jobs").mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict[str, object]:
        """Safe, secret-free configuration summary for logs and /health."""
        return {
            "version": self.version,
            "ocr_engine": self.ocr_engine,
            "paddle_lang": self.paddle_lang,
            "paddle_profile": self.paddle_profile,
            "paddle_det_limit_side_len": self.paddle_det_limit_side_len,
            "inpaint_engine": self.inpaint_engine,
            "lama_model": str(self.resolved_lama_path),
            "lama_model_exists": self.lama_model_available,
            "lama_model_source": self.lama_model_source,
            "lama_precision": self.lama_precision,
            "lama_device": self.lama_device,
            "lama_auto_download": self.lama_auto_download,
            "lama_threads": self.lama_threads,
            "inpaint_fast_stddev": self.inpaint_fast_stddev,
            "inpaint_roi_padding": self.inpaint_roi_padding,
            "inpaint_max_roi_patches": self.inpaint_max_roi_patches,
            "progressive_preview": self.progressive_preview,
            "translation_provider": self.effective_translation_provider,
            "translation_configured": self.translation_configured,
            "deepseek_model": self.deepseek_model,
            "queue_workers": self.queue_workers,
            "max_image_side": self.max_image_side,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
    os.environ.pop("OTR_SETTINGS_OVERRIDE", None)
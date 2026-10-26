"""Shared desktop-client configuration and paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path


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


PROJECT_ROOT = _app_root()
BUNDLE_ROOT = _bundle_root()
APP_NAME = "O-Translator"
API_URL = os.environ.get("OTR_API_URL", "http://127.0.0.1:8765").rstrip("/")
AUTOSTART = os.environ.get("OTR_CLIENT_AUTOSTART_BACKEND", "true").lower() not in {
    "0",
    "false",
    "no",
}
DEFAULT_HOTKEY = os.environ.get("OTR_HOTKEY", "f2").strip().lower()
HOTKEY_SETTINGS_VERSION = "2"
LEGACY_DEFAULT_HOTKEY = "ctrl+alt+t"
CAPTURE_DIR = PROJECT_ROOT / "data" / "client-captures"
BACKEND_LOG = PROJECT_ROOT / "data" / "desktop-backend.log"
ICON_PATH = BUNDLE_ROOT / "client" / "web" / "assets" / "favicon.svg"


def ensure_capture_dir() -> Path:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    return CAPTURE_DIR


def mime_for(path: Path) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(path.suffix.lower(), "application/octet-stream")


def is_image_path(value: str) -> bool:
    return Path(value).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
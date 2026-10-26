"""Qt stylesheet loader shared by the desktop windows."""

from __future__ import annotations

from pathlib import Path

STYLE_DIR = Path(__file__).resolve().parent
if not (STYLE_DIR / "base.qss").is_file():  # packaged builds keep data outside the module tree
    from client.config import BUNDLE_ROOT

    bundled = BUNDLE_ROOT / "client" / "ui"
    if (bundled / "base.qss").is_file():
        STYLE_DIR = bundled
BASE_QSS = STYLE_DIR / "base.qss"
EXTRA_QSS = STYLE_DIR / "extra.qss"


def load_stylesheet() -> str:
    parts = []
    for path in (BASE_QSS, EXTRA_QSS):
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


APP_STYLE = load_stylesheet()
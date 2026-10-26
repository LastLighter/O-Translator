"""Render ``client/web/assets/favicon.svg`` into the Windows ``.ico`` used by
the packaged client.

Usage: python scripts/make_icon.py [output.ico]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "client" / "web" / "assets" / "favicon.svg"
DEFAULT_TARGET = ROOT / "build" / "icon.ico"
SIZES = (16, 24, 32, 48, 64, 128, 256)


def render_png(size: int, target: Path) -> None:
    from PySide6 import QtCore, QtGui, QtSvg

    renderer = QtSvg.QSvgRenderer(str(SOURCE))
    if not renderer.isValid():
        raise RuntimeError(f"Cannot parse SVG: {SOURCE}")
    image = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    if not image.save(str(target), "PNG"):
        raise RuntimeError(f"Could not write {target}")


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else DEFAULT_TARGET
    target.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6 import QtGui

    app = QtGui.QGuiApplication([])  # noqa: F841 - required for QPainter
    frames = []
    for size in SIZES:
        png = target.with_name(f"icon-{size}.png")
        render_png(size, png)
        frames.append(png)

    from PIL import Image

    images = [Image.open(png).convert("RGBA") for png in frames]
    images[-1].save(target, format="ICO", sizes=[(s, s) for s in SIZES])
    for png in frames:
        png.unlink(missing_ok=True)
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

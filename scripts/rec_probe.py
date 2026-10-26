"""Probe PaddleOCR recognition models on fixed manga crops.

Detection is skipped: the crops and their reading orientation are known, so this
isolates the recognizer's ability to read brush fonts, long-vowel marks, and
tiny subtitle lines.

Usage::

    python scripts/rec_probe.py data/repro/original-jp.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# (label, bbox, vertical)
CROPS = [
    ("title-a", (384, 14, 448, 212), True),
    ("title-b", (330, 32, 402, 428), True),
    ("mono-a", (68, 254, 96, 400), True),
    ("mono-b", (48, 254, 76, 504), True),
    ("mono-c", (30, 254, 60, 478), True),
    ("subtitle-1", (168, 620, 232, 636), False),
    ("subtitle-2", (168, 630, 232, 650), False),
    ("subtitle-3", (168, 642, 236, 664), False),
    ("bottom-small", (12, 604, 172, 638), False),
    ("bottom-big", (8, 630, 172, 700), False),
    ("bottom-end", (148, 648, 214, 692), False),
]

_MODELS: dict[str, object] = {}


def model(name: str):
    if name not in _MODELS:
        from paddleocr import TextRecognition

        _MODELS[name] = TextRecognition(model_name=name)
    return _MODELS[name]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--scale", type=float, action="append", default=None)
    parser.add_argument("--rec", action="append", default=None)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    scales = args.scale or [1.0, 2.0, 3.0]
    recs = args.rec or [
        "PP-OCRv5_mobile_rec",
        "PP-OCRv5_server_rec",
        "japan_PP-OCRv3_mobile_rec",
    ]
    for rec_name in recs:
        print(f"\n########## {rec_name} ##########")
        try:
            recognizer = model(rec_name)
        except Exception as exc:
            print("INIT ERROR", type(exc).__name__, exc)
            continue
        for label, (x0, y0, x1, y1), vertical in CROPS:
            print(f"\n--- {label} {(x0, y0, x1, y1)} vertical={vertical}")
            for scale in scales:
                for bgr in (True, False):
                    crop = image[y0:y1, x0:x1]
                    if not bgr:
                        crop = np.ascontiguousarray(crop[:, :, ::-1])
                    if scale != 1.0:
                        crop = cv2.resize(
                            crop,
                            (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
                            interpolation=cv2.INTER_CUBIC,
                        )
                    if vertical:
                        crop = np.ascontiguousarray(np.rot90(crop, k=1))
                    item = recognizer.predict(crop)[0]
                    text = str(item.get("rec_text") or "")
                    score = float(item.get("rec_score") or 0.0)
                    print(f"  scale={scale:g} bgr={bgr} {score:.3f} {text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Probe PaddleOCR detector/recognizer variants on one image.

Development helper used to tune manga OCR. Runs detection at several scales and
recognition with different models / channel orders, printing every line so the
results can be compared without touching the production code.

Usage::

    python scripts/ocr_probe.py data/repro/original-jp.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


_MODELS: dict[tuple[str, str], object] = {}


def _model(kind: str, name: str, **kwargs):
    key = (kind, name)
    if key not in _MODELS:
        from paddleocr import TextDetection, TextRecognition

        cls = TextDetection if kind == "det" else TextRecognition
        _MODELS[key] = cls(model_name=name, **kwargs)
    return _MODELS[key]


def recognize(
    image_bgr: np.ndarray,
    *,
    det_name: str,
    rec_name: str,
    scale: float,
    limit_side_len: int = 2560,
) -> list[dict]:
    height, width = image_bgr.shape[:2]
    resized = image_bgr
    if scale != 1.0:
        resized = cv2.resize(
            image_bgr,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    detector = _model("det", det_name, limit_side_len=limit_side_len)
    recognizer = _model("rec", rec_name)
    raw = detector.predict(resized)
    result = raw[0] if raw else {}
    polygons = result.get("dt_polys", []) if hasattr(result, "get") else []
    out: list[dict] = []
    for polygon in polygons:
        points = np.asarray(polygon).reshape(-1, 2)
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        pad = max(2, round(4 * scale))
        cx0, cy0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
        cx1 = min(resized.shape[1], int(x1) + pad)
        cy1 = min(resized.shape[0], int(y1) + pad)
        crop = resized[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            continue
        crop_h, crop_w = crop.shape[:2]
        vertical = crop_h > crop_w * 1.25
        if vertical:
            crop = np.ascontiguousarray(np.rot90(crop, k=1))
        item = recognizer.predict(crop)[0]
        text = str(item.get("rec_text") or "")
        score = float(item.get("rec_score") or 0.0)
        out.append(
            {
                "text": text,
                "score": round(score, 4),
                "bbox": [
                    round(x0 / scale),
                    round(y0 / scale),
                    round((x1 - x0) / scale),
                    round((y1 - y0) / scale),
                ],
                "vertical": vertical,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--scale", type=float, action="append", default=None)
    parser.add_argument("--rec", action="append", default=None)
    parser.add_argument("--det", default="PP-OCRv5_server_det")
    parser.add_argument("--bgr", action="store_true", default=False)
    parser.add_argument("--rgb", action="store_true")
    args = parser.parse_args()

    rgb = np.asarray(cv2.imread(str(args.image))[:, :, ::-1])
    scales = args.scale or [1.0, 2.0]
    recs = args.rec or ["PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec"]
    for bgr in ([True] if args.bgr else [False] if args.rgb else [True, False]):
        image = np.ascontiguousarray(rgb[:, :, ::-1]) if bgr else rgb
        for rec in recs:
            for scale in scales:
                print(f"\n===== bgr={bgr} rec={rec} scale={scale} =====")
                try:
                    rows = recognize(
                        image, det_name=args.det, rec_name=rec, scale=scale
                    )
                except Exception as exc:  # pragma: no cover - model availability
                    print("ERROR", type(exc).__name__, exc)
                    continue
                for row in rows:
                    print(
                        f"{row['bbox']} v={row['vertical']} {row['score']:.3f} "
                        f"{row['text']}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

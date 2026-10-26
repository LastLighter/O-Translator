"""Probe tiny subtitle crops with preprocessing variants."""

from __future__ import annotations

import sys

import cv2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def variants(crop):
    yield "orig", crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    yield "gray", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(gray)
    yield "clahe", cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    yield "sharp", cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "otsu", cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 6
    )
    yield "adap", cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)


def main() -> int:
    from paddleocr import TextRecognition

    image = cv2.imread("data/repro/original-jp.jpg")
    recognizer = TextRecognition(model_name="PP-OCRv5_mobile_rec")
    boxes = [
        ("l1", (165, 617, 240, 634)),
        ("l2", (165, 632, 240, 649)),
        ("l3", (165, 647, 240, 665)),
        ("all", (165, 617, 240, 665)),
    ]
    for name, (x0, y0, x1, y1) in boxes:
        crop = image[y0:y1, x0:x1]
        print("==", name)
        for variant_name, variant in variants(crop):
            for scale in (3, 5, 7):
                upscaled = cv2.resize(
                    variant, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )
                item = recognizer.predict(upscaled)[0]
                print(
                    f"  {variant_name} x{scale}: "
                    f"{float(item.get('rec_score') or 0):.3f} "
                    f"{item.get('rec_text')!r}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

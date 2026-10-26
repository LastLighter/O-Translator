"""Re-render an existing repro result without re-running OCR/inpainting.

Usage::

    python scripts/render_probe.py data/repro/run-v5 --out data/repro/render-probe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_pipeline.types import TextRegion
from core_pipeline.typesetting import FontBook, Renderer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/repro/render-probe"))
    parser.add_argument("--mode", default="manga")
    parser.add_argument("--target-lang", default="zh")
    args = parser.parse_args()

    data = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    original = Image.open(args.run / "original.png").convert("RGB")
    clean = Image.open(args.run / "clean.png").convert("RGB")
    regions: list[TextRegion] = []
    translations: dict[int, str] = {}
    for row in data["regions"]:
        box = row["bbox"]
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        regions.append(
            TextRegion(
                id=row["id"],
                text=row["text"],
                confidence=row.get("confidence", 0.9),
                polygon=[
                    (x, y),
                    (x + w, y),
                    (x + w, y + h),
                    (x, y + h),
                ],
                direction=row["direction"],
            )
        )
        translations[row["id"]] = row.get("translation") or row["text"]

    renderer = Renderer(FontBook())
    output = renderer.render(
        original,
        clean,
        regions,
        translations,
        target_lang=args.target_lang,
        mode=args.mode,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    output.image.save(args.out / "result.png")
    for block in output.blocks:
        print(
            json.dumps(
                {
                    "id": block.id,
                    "bbox": block.bbox,
                    "font": block.font_size,
                    "weight": block.font_weight,
                    "fill": block.text_color,
                    "stroke": block.stroke_color,
                    "stroke_w": block.stroke_width,
                    "lines": block.lines,
                    "overflow": block.overflow,
                },
                ensure_ascii=False,
            )
        )
    print("warnings", output.warnings)
    print(f"saved -> {args.out / 'result.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

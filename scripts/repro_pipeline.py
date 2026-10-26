"""Run the full pipeline once on a local image and dump every stage.

This is a development helper for tuning OCR, translation, inpainting, and
typesetting against real pages. It intentionally bypasses the HTTP server so a
single iteration only pays for the model calls that matter.

Usage::

    python scripts/repro_pipeline.py data/repro/original-jp.jpg --out data/repro/run1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load_windows_api_key() -> None:
    """Mirror the desktop launcher: pick up the persisted user key."""
    if os.environ.get("DEEPSEEK_API_KEY"):
        return
    if sys.platform != "win32":
        return
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, "DEEPSEEK_API_KEY")
    except OSError:
        return
    if isinstance(value, str) and value.strip():
        os.environ["DEEPSEEK_API_KEY"] = value.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/repro/run"))
    parser.add_argument("--source-lang", default="ja")
    parser.add_argument("--target-lang", default="zh")
    parser.add_argument("--mode", default="manga")
    parser.add_argument("--reading-order", default="auto")
    parser.add_argument("--no-translate", action="store_true")
    parser.add_argument("--font-scale", type=float, default=1.0)
    args = parser.parse_args()

    _load_windows_api_key()

    from backend.config import Settings
    from backend.main import _build_pipeline
    from core_pipeline.types import PipelineOptions

    settings = Settings()
    issues: list[str] = []
    pipeline = _build_pipeline(settings, issues)
    options = PipelineOptions(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        mode=args.mode,
        reading_order=args.reading_order,
        font_scale=args.font_scale,
        translate=not args.no_translate,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    payload = args.image.read_bytes()

    async def _run():
        try:
            return await pipeline.run(payload, options)
        finally:
            await pipeline.aclose()

    result = asyncio.run(_run())

    result.original.save(args.out / "original.png")
    result.clean.save(args.out / "clean.png")
    result.mask.save(args.out / "mask.png")
    result.rendered.save(args.out / "result.png")

    print(f"image={args.image} source={result.source_size} processed={result.processed_size}")
    print(f"ocr={result.ocr_engine} inpaint={result.inpaint_engine} "
          f"translator={result.translation_provider}/{result.translation_model}")
    print(f"timings={result.timings_ms}")
    print("warnings=", json.dumps(result.warnings, ensure_ascii=False, indent=2))
    print("---- regions ----")
    for region in result.regions:
        print(
            json.dumps(
                {
                    "id": region.id,
                    "text": region.text,
                    "confidence": region.confidence,
                    "bbox": region.bbox.model_dump(),
                    "direction": region.direction,
                    "translation": result.translations.get(region.id),
                },
                ensure_ascii=False,
            )
        )
    print("---- blocks ----")
    for block in result.blocks:
        print(
            json.dumps(
                {
                    "id": block.id,
                    "source": block.source_text,
                    "translated": block.translated_text,
                    "bbox": block.bbox,
                    "layout": block.layout,
                    "font_size": block.font_size,
                    "lines": block.lines,
                    "overflow": block.overflow,
                    "text_color": block.text_color,
                    "bg": block.background_color,
                },
                ensure_ascii=False,
            )
        )
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "id": region.id,
                        "text": region.text,
                        "confidence": region.confidence,
                        "bbox": region.bbox.model_dump(),
                        "direction": region.direction,
                        "translation": result.translations.get(region.id),
                    }
                    for region in result.regions
                ],
                "blocks": [block.model_dump() for block in result.blocks],
                "warnings": result.warnings,
                "timings": result.timings_ms,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

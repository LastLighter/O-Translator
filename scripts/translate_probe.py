"""Translate an existing region dump with the production prompt.

Development helper: reads ``summary.json`` from a repro run (or a plain JSON
list of ``{text, bbox, direction}`` objects), calls the configured translator,
and prints each source/translation pair. It skips OCR and inpainting so prompt
iterations take a few seconds instead of a minute.

Usage::

    python scripts/translate_probe.py data/repro/run-baseline-ocr/summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load_windows_api_key() -> None:
    if os.environ.get("DEEPSEEK_API_KEY") or sys.platform != "win32":
        return
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, "DEEPSEEK_API_KEY")
    except OSError:
        return
    if isinstance(value, str) and value.strip():
        os.environ["DEEPSEEK_API_KEY"] = value.strip()


async def main_async(args: argparse.Namespace) -> int:
    from backend.config import Settings
    from core_pipeline.translation import build_translator
    from core_pipeline.types import PipelineOptions, TextRegion

    settings = Settings()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data.get("regions", data) if isinstance(data, dict) else data
    regions: list[TextRegion] = []
    for index, row in enumerate(rows):
        bbox = row.get("bbox") or {}
        if isinstance(bbox, list):
            x, y, w, h = bbox
        else:
            x = int(bbox.get("x", 0))
            y = int(bbox.get("y", 0))
            w = int(bbox.get("w", 10))
            h = int(bbox.get("h", 10))
        regions.append(
            TextRegion(
                id=index,
                text=row["text"],
                polygon=[
                    (x, y),
                    (x + w, y),
                    (x + w, y + h),
                    (x, y + h),
                ],
                direction=row.get("direction", "horizontal"),
            )
        )
    options = PipelineOptions(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        mode=args.mode,
        reading_order=args.reading_order,
    )
    translator = build_translator(
        provider="deepseek",
        api_key=settings.deepseek_api_key,
        model=settings.deepseek_model,
        base_url=settings.deepseek_base_url,
        timeout_s=settings.translation_timeout_s,
        batch_size=settings.translation_batch_size,
        concurrency=1,
        max_retries=settings.translation_max_retries,
    )
    output = await translator.translate(regions, options)
    await translator.aclose()
    print(f"model={output.model} tokens={output.usage} elapsed={output.elapsed_ms:.0f}ms")
    for region in regions:
        print(f"[{region.id}] {region.text}")
        print(f"    -> {output.translations.get(region.id)}")
    print("warnings=", json.dumps(output.warnings, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--source-lang", default="ja")
    parser.add_argument("--target-lang", default="zh")
    parser.add_argument("--mode", default="manga")
    parser.add_argument("--reading-order", default="auto")
    args = parser.parse_args()
    _load_windows_api_key()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

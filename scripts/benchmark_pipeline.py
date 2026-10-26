"""Warm-model benchmark for OCR and typesetting on real screenshots.

Run from the repository root::

    python scripts/benchmark_pipeline.py
    python scripts/benchmark_pipeline.py --source-lang auto --mode auto
    python scripts/benchmark_pipeline.py --repeats 5 --images examples/demo_manga.png

The OCR models are warmed before timing so the numbers describe steady-state
jobs, not the first launch. Use ``--cold`` to measure a fresh process' first
read (this loads the PaddleOCR models and takes several seconds).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from functools import partial
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core_pipeline.ocr import build_ocr_backend  # noqa: E402
from core_pipeline.types import OcrOutput  # noqa: E402
from core_pipeline.typesetting import FontBook, Renderer  # noqa: E402


def default_images() -> list[Path]:
    images = sorted((ROOT / "examples").glob("*.png"))
    images += sorted((ROOT / "data" / "jobs").glob("*/original.png"))
    return [path for path in images if path.is_file()]


def time_call(function, *, repeats: int) -> tuple[float, object]:
    samples: list[float] = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples), result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Warm-model benchmark for OCR and typesetting on real screenshots. "
            "The OCR models are warmed before timing so the numbers describe "
            "steady-state jobs, not the first launch."
        ),
        epilog=(
            "Run from the repository root. Use --cold to also measure a fresh "
            "process' first read (loads the PaddleOCR models, several seconds)."
        ),
    )
    parser.add_argument("--images", nargs="*", type=Path, default=None)
    parser.add_argument("--source-lang", default="ja")
    parser.add_argument("--mode", default="manga")
    parser.add_argument("--target-lang", default="zh")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cold", action="store_true", help="also time a cold first read")
    parser.add_argument("--no-render", action="store_true")
    arguments = parser.parse_args()

    images = arguments.images or default_images()
    if not images:
        print("No images found.", file=sys.stderr)
        return 1

    backend = build_ocr_backend(
        engine="paddle",
        paddle_lang="ch",
        paddle_auto_fallback=True,
    )
    warm_started = time.perf_counter()
    warm_up = getattr(backend, "warm_up", None)
    if callable(warm_up):
        warm_up()
    print(f"OCR warm-up: {(time.perf_counter() - warm_started):.1f}s\n")

    if arguments.cold:
        cold_backend = build_ocr_backend(
            engine="paddle",
            paddle_lang="ch",
            paddle_auto_fallback=True,
        )
        first = Image.open(images[0]).convert("RGB")
        started = time.perf_counter()
        cold_backend.read(first, source_lang=arguments.source_lang, mode=arguments.mode)
        print(f"Cold first read ({images[0].name}): {(time.perf_counter() - started) * 1000:.0f}ms\n")

    renderer = Renderer(FontBook())
    print(
        f"{'image':<36} {'size':>11} {'regions':>7} {'ocr ms':>9} "
        f"{'render ms':>10} {'total ms':>9}"
    )
    print("-" * 88)
    for path in images:
        image = Image.open(path).convert("RGB")
        ocr_ms, output = time_call(
            partial(
                backend.read,
                image,
                source_lang=arguments.source_lang,
                mode=arguments.mode,
            ),
            repeats=max(1, arguments.repeats),
        )
        assert isinstance(output, OcrOutput)
        render_ms = 0.0
        if not arguments.no_render and output.regions:
            clean = image.copy()
            translations = {region.id: region.text for region in output.regions}
            render_ms, _ = time_call(
                partial(
                    renderer.render,
                    image,
                    clean,
                    output.regions,
                    translations,
                    target_lang=arguments.target_lang,
                    mode=arguments.mode,
                ),
                repeats=max(1, arguments.repeats),
            )
        print(
            f"{path.name:<36} {image.width}x{image.height:<6} {len(output.regions):>7} "
            f"{ocr_ms:>9.0f} {render_ms:>10.1f} {ocr_ms + render_ms:>9.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
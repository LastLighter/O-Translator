"""Verify the LaMa CUDA path and compare it with CPU output.

Run from the repository root after installing the GPU add-on or the
``onnxruntime-gpu`` + NVIDIA runtime wheels:

    python scripts/verify_gpu.py
    python scripts/verify_gpu.py --require-gpu
    python scripts/verify_gpu.py --model models/lama/lama_fp32_sim.onnx

The script activates an optional ``gpu_runtime/`` add-on first, prints the
selected providers, then benchmarks warm CPU and CUDA sessions on the same
512x512 input and checks that the pixel difference stays below ``--max-diff``.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core_pipeline.runtime import (  # noqa: E402
    activate_gpu_runtime,
    available_providers,
    create_session,
    describe_runtime,
)


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    sample = ROOT / "examples" / "demo_manga.png"
    if sample.is_file():
        import cv2
        from PIL import Image

        rgb = np.asarray(Image.open(sample).convert("RGB"))
        rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        image = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        mask = np.zeros((1, 1, 512, 512), dtype=np.float32)
        mask[:, :, 120:400, 100:430] = 1.0
        return image, mask

    rng = np.random.default_rng(0)
    image = rng.random((1, 3, 512, 512), dtype=np.float32)
    mask = np.zeros((1, 1, 512, 512), dtype=np.float32)
    mask[:, :, 96:360, 96:420] = 1.0
    return image, mask


def _run(session, image: np.ndarray, mask: np.ndarray, repeats: int) -> tuple[np.ndarray, list[float]]:
    inputs = session.get_inputs()
    feeds = {inputs[0].name: image, inputs[1].name: mask}
    output = None
    samples: list[float] = []
    for index in range(repeats + 1):
        started = time.perf_counter()
        output = session.run(None, feeds)[0]
        elapsed = (time.perf_counter() - started) * 1000.0
        if index:
            samples.append(elapsed)
    assert output is not None
    return np.asarray(output, dtype=np.float32), samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "lama" / "lama_fp32_sim.onnx")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-diff", type=float, default=5.0)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    activate_gpu_runtime()
    providers = available_providers()
    print(f"available providers: {providers}")
    print(f"runtime: {describe_runtime()}")
    if "CUDAExecutionProvider" not in providers:
        print("CUDAExecutionProvider is not available.", file=sys.stderr)
        return 1 if args.require_gpu else 0
    if not args.model.is_file():
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 1

    image, mask = _inputs()
    cpu_session, cpu_providers = create_session(str(args.model), device="cpu")
    cuda_session, cuda_providers = create_session(str(args.model), device="cuda")
    print(f"CPU session providers: {cpu_providers}")
    print(f"CUDA session providers: {cuda_providers}")
    if "CUDAExecutionProvider" not in cuda_providers:
        print("CUDA session fell back to CPU.", file=sys.stderr)
        return 1

    cpu_output, cpu_samples = _run(cpu_session, image, mask, args.repeats)
    cuda_output, cuda_samples = _run(cuda_session, image, mask, args.repeats)
    diff = np.abs(cpu_output - cuda_output)
    cpu_median = statistics.median(cpu_samples)
    cuda_median = statistics.median(cuda_samples)
    print(f"CPU warm:  {cpu_median:.1f} ms")
    print(f"CUDA warm: {cuda_median:.1f} ms ({cpu_median / max(1e-6, cuda_median):.1f}x)")
    print(
        f"output diff: max {diff.max():.4f}, mean {diff.mean():.6f}, "
        f"pixels > {args.max_diff:g}: {int((diff > args.max_diff).sum())}"
    )
    if diff.max() > args.max_diff:
        print("GPU output differs from CPU beyond the threshold.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
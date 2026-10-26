"""Assemble the optional CUDA ONNX Runtime add-on for the portable build.

The base O-Translator zip deliberately stays CPU-only: a CUDA runtime adds
roughly 2 GB of cuBLAS/cuDNN/cuFFT/cuRAND DLLs. This script copies the installed
``onnxruntime-gpu`` package plus its CUDA dependencies into a self-contained
``gpu_runtime/`` folder and zips it. Extracting the add-on next to
``O-Translator.exe`` makes ``core_pipeline.runtime.activate_gpu_runtime`` prefer
CUDA automatically; without it the base package keeps using CPU ORT.

Usage:
    python scripts/setup_gpu_runtime.ps1       # installs .venv-gpu-cuda
    python scripts/build_gpu_runtime.py         # writes dist/O-Translator-gpu-runtime-cuda12.zip
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SITE_PACKAGES = ROOT / ".venv-gpu-cuda" / "Lib" / "site-packages"
DEFAULT_OUTPUT = ROOT / "dist" / "O-Translator-gpu-runtime-cuda12.zip"
BUILD_ROOT = ROOT / "build" / "gpu_runtime_pack"
ROOT_NAME = "gpu_runtime"

README = """O-Translator CUDA runtime add-on
================================

Extract this archive next to O-Translator.exe so the layout is:

    O-Translator.exe
    gpu_runtime/
      onnxruntime/...
      README-GPU.txt

The application auto-detects gpu_runtime/ at startup and prefers
CUDAExecutionProvider. If the NVIDIA driver does not support the bundled CUDA
12.6 runtime, the existing CPU ONNX Runtime is used instead (no crash).

Requirements:
  * Windows x64
  * NVIDIA GPU + driver with CUDA 12.x support (RTX 20-series or newer)
  * ~2.5 GB free disk space

You can also point OTR_GPU_RUNTIME_DIR at this folder manually, or set
OTR_DISABLE_GPU=1 to force CPU inference.

Verified on an RTX 5060 Laptop GPU: LaMa 512x512 warm inference ~0.21-0.30 s
(CPU: ~2.1 s), and RapidOCR PP-OCR detection + recognition ~55-125 ms for
900x1280 manga / 1400x900 UI pages (CPU: several times slower). The model
weights themselves are already in the base package; PP-OCRv6 small ships inside
the rapidocr wheel.
"""


def _copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise SystemExit(f"Missing source directory: {source}")
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        dirs_exist_ok=True,
    )


def _copy_cuda_runtime(site_packages: Path, target_capi: Path) -> int:
    """Place CUDA/cuDNN DLLs beside the ORT provider DLLs.

    ``onnxruntime_providers_cuda.dll`` resolves its dependencies with the
    default Windows loader. Keeping the runtime DLLs in ``capi`` (and putting
    that directory on PATH) avoids relying on a system-wide CUDA toolkit.
    """
    count = 0
    nvidia_root = site_packages / "nvidia"
    if not nvidia_root.is_dir():
        return 0
    patterns = ("cublas*.dll", "cudnn*.dll", "cudart*.dll", "cufft*.dll", "curand*.dll", "nvblas*.dll")
    for folder in nvidia_root.iterdir():
        bin_dir = folder / "bin"
        if not bin_dir.is_dir():
            continue
        for pattern in patterns:
            for dll in bin_dir.glob(pattern):
                shutil.copy2(dll, target_capi / dll.name)
                count += 1
    return count


def _zip_tree(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for path in sorted(source.rglob("*")):
            arcname = (Path(ROOT_NAME) / path.relative_to(source)).as_posix()
            if path.is_dir():
                if not any(path.iterdir()):
                    archive.writestr(arcname + "/", "")
            else:
                archive.write(path, arcname)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", type=Path, default=DEFAULT_SITE_PACKAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args(argv[1:])

    site_packages = args.site_packages.resolve()
    source_ort = site_packages / "onnxruntime"
    if not (source_ort / "capi" / "onnxruntime_providers_cuda.dll").is_file():
        raise SystemExit(
            f"{source_ort} is not a CUDA onnxruntime build. Run "
            "scripts/setup_gpu_runtime.ps1 first."
        )

    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    pack_root = BUILD_ROOT / ROOT_NAME
    pack_root.mkdir(parents=True)

    print(f"Copying {source_ort} ...")
    _copy_tree(source_ort, pack_root / "onnxruntime")
    cuda_dlls = _copy_cuda_runtime(site_packages, pack_root / "onnxruntime" / "capi")
    if cuda_dlls == 0 and not list((pack_root / "onnxruntime" / "capi").glob("cudnn*.dll")):
        raise SystemExit(
            "No CUDA/cuDNN DLLs found. Install the nvidia-*-cu12 wheels in the "
            "GPU venv (scripts/setup_gpu_runtime.ps1)."
        )
    (pack_root / "README-GPU.txt").write_text(README, encoding="utf-8")

    total = sum(path.stat().st_size for path in pack_root.rglob("*") if path.is_file())
    print(f"Prepared {pack_root} ({total / 1024 / 1024:.0f} MB, {cuda_dlls} CUDA DLLs)")

    if args.no_zip:
        return 0
    print(f"Zipping to {args.output} ...")
    _zip_tree(pack_root, args.output.resolve())
    size = args.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {args.output} ({size:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the O-Translator Windows desktop client.

Build (see scripts/build_windows_exe.ps1):

    python -m PyInstaller --noconfirm --clean O-Translator.spec

Set ``OTR_BUNDLE_OCR=0`` for a quick UI-only smoke build without OCR.
"""

import importlib.util
import os
import re
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    copy_metadata,
)

BUNDLE_OCR = os.environ.get(
    "OTR_BUNDLE_OCR", os.environ.get("OTR_BUNDLE_PADDLE", "1")
).strip().lower() not in {"0", "false", "no"}

datas = [
    ("client/web", "client/web"),
    ("client/ui/base.qss", "client/ui"),
    ("client/ui/extra.qss", "client/ui"),
    ("models/README.md", "models"),
]

# Ship the already-folded LaMa graph so a freshly extracted build never has to
# download ~200 MB on first launch. Packaged data lives under ``_internal`` in a
# PyInstaller onedir build; backend/config.py searches both the exe directory
# (writable, for downloads) and ``_MEIPASS`` (bundled, read-only).
_lama_bundle = os.environ.get(
    "OTR_LAMA_BUNDLE_MODEL", "models/lama/lama_fp32_sim.onnx"
)
if not os.path.isfile(_lama_bundle):
    _lama_fallback = "models/lama/lama_fp32.onnx"
    if os.path.isfile(_lama_fallback):
        _lama_bundle = _lama_fallback
if os.path.isfile(_lama_bundle):
    datas.append((_lama_bundle, "models/lama"))
    print(f"[spec] bundling LaMa model: {_lama_bundle}")
else:
    print("[spec] WARNING: no LaMa ONNX model found; first launch will download it")

# PP-OCRv4/v5/v6 graphs downloaded by scripts/download_models.py --ocr are
# bundled next to the LaMa weights. The v6 small graphs ship inside the
# rapidocr wheel and are collected below.
_ocr_models_dir = "models/rapidocr"
if BUNDLE_OCR and os.path.isdir(_ocr_models_dir):
    _ocr_files = sorted(
        name
        for name in os.listdir(_ocr_models_dir)
        if name.endswith(".onnx")
        # PP-OCRv6 small already ships inside the rapidocr wheel.
        and not (name.startswith("PP-OCRv6_") and name.endswith("_small.onnx"))
    )
    if _ocr_files:
        for _name in _ocr_files:
            datas.append(
                (os.path.join(_ocr_models_dir, _name), _ocr_models_dir)
            )
        print(f"[spec] bundling {len(_ocr_files)} RapidOCR model(s)")

binaries = []
hiddenimports = [
    # uvicorn resolves these protocol/loop implementations dynamically.
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "anyio._backends._asyncio",
    # Used only for QIcon(SVG) and Windows shell integration.
    "PySide6.QtSvg",
    # Single-instance lock/notification uses QLocalServer/QLocalSocket.
    "PySide6.QtNetwork",
    # Automatic window/control picking. These are ordinary imports from
    # client.main, but listing them keeps the frozen module graph explicit.
    "client.services.target_detector",
    "client.services.ui_target",
    "client.services.uia_targets",
    "client.services.win32_targets",
]

if BUNDLE_OCR:
    # RapidOCR imports its ONNX Runtime engine lazily, so PyInstaller's static
    # walk cannot see it. Collect only the ONNX Runtime backend: the torch,
    # TensorRT, OpenVINO, Paddle, and MNN engines stay out of the build.
    hiddenimports += [
        "rapidocr",
        "rapidocr.main",
        "rapidocr.ch_ppocr_det",
        "rapidocr.ch_ppocr_rec",
        "rapidocr.inference_engine.base",
        "rapidocr.inference_engine.onnxruntime",
        "rapidocr.inference_engine.onnxruntime.main",
        "rapidocr.inference_engine.onnxruntime.provider_config",
        "rapidocr.utils.download_file",
        "rapidocr.utils.parse_parameters",
        "rapidocr.utils.typings",
        "omegaconf",
        "yaml",
        "shapely",
        "pyclipper",
        "requests",
        "tqdm",
        "colorlog",
    ]
    datas += collect_data_files(
        "rapidocr",
        includes=["*.yaml", "models/*.onnx", "models/*.txt"],
    )
    # Tiny .dist-info folders keep --doctor and support logs informative.
    for _dist in ("rapidocr", "onnxruntime"):
        try:
            datas += copy_metadata(_dist)
        except Exception as _exc:  # pragma: no cover - build-time diagnostic
            print(f"[spec] cannot copy metadata for {_dist}: {_exc}")
    # The base package stays CPU-only: ONNX Runtime's CUDA/cuDNN provider
    # DLLs belong to the optional gpu_runtime/ add-on, not the main zip.
    _cuda_dll = re.compile(
        r"^(cublas|cublasLt|cudart|cudnn|cufft|cufftw|curand|nvblas|"
        r"onnxruntime_providers_(cuda|tensorrt))",
        re.IGNORECASE,
    )
    for _src, _dst in collect_dynamic_libs("onnxruntime"):
        if _cuda_dll.search(Path(_src).name):
            continue
        binaries.append((_src, _dst))
    _ort_spec = importlib.util.find_spec("onnxruntime")
    if _ort_spec is not None and _ort_spec.origin:
        _pybind = (
            Path(_ort_spec.origin).parent
            / "capi"
            / "onnxruntime_pybind11_state.pyd"
        )
        if _pybind.is_file():
            binaries.append((str(_pybind), "onnxruntime/capi"))
    hiddenimports += [
        "onnxruntime",
        "onnxruntime.capi._pybind_state",
        "onnxruntime.capi._ld_preload",
        "onnxruntime.capi.onnxruntime_inference_collection",
        "onnxruntime.capi.onnxruntime_validation",
    ]

excluded_ocr_modules = [
    "paddle",
    "paddleocr",
    "paddlex",
    "torch",
    "torchvision",
    "tensorrt",
    "openvino",
    "MNN",
    "scipy",
    "sklearn",
    "matplotlib",
    "rapidocr.inference_engine.pytorch",
    "rapidocr.inference_engine.tensorrt",
    "rapidocr.inference_engine.paddle",
    "rapidocr.inference_engine.openvino",
    "rapidocr.inference_engine.mnn",
]

a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "pytest",
    ]
    + excluded_ocr_modules,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="O-Translator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon="build/icon.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="O-Translator",
)
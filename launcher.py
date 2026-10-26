"""PyInstaller entry point for the packaged Windows desktop client.

Two modes share one executable:

* ``O-Translator.exe``             -> PySide6 desktop UI
* ``O-Translator.exe --backend``   -> local FastAPI service (spawned by the UI)

Running the backend from the same exe keeps the distribution to a single
binary and avoids depending on a system Python installation.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import sys


def _ensure_std_streams() -> None:
    """Repair stdio for a windowed PyInstaller build.

    With ``console=False`` the bootloader may set ``sys.stdout``/``sys.stderr``
    to ``None`` even though the parent process redirected the underlying file
    descriptors. Logging (uvicorn, the pipeline) needs usable streams, so
    re-open the inherited descriptors and force UTF-8 to avoid console-codec
    crashes on translated text.
    """
    for name, fd in (("stdout", 1), ("stderr", 2)):
        stream = getattr(sys, name, None)
        if stream is None:
            try:
                stream = open(fd, "w", encoding="utf-8", errors="replace", closefd=False)
            except OSError:
                stream = open(os.devnull, "w", encoding="utf-8")
            setattr(sys, name, stream)
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(OSError, ValueError):
                reconfigure(encoding="utf-8", errors="replace")


def _run_backend() -> int:
    _ensure_std_streams()
    from core_pipeline.runtime import activate_gpu_runtime

    activate_gpu_runtime()
    from backend.main import run

    run()
    return 0


def _run_gui(show_window: bool = False) -> int:
    _ensure_std_streams()
    from client.main import main

    return main(show_window=show_window)


def _run_doctor() -> int:
    """Diagnose a packaged build: metadata, imports, and RapidOCR init."""
    _ensure_std_streams()
    from core_pipeline.runtime import activate_gpu_runtime

    activate_gpu_runtime()
    import importlib.metadata
    import traceback

    print(f"frozen={getattr(sys, 'frozen', False)}")
    print(f"meipass={getattr(sys, '_MEIPASS', '')}")
    for dep in (
        "rapidocr",
        "onnxruntime",
        "opencv-python-headless",
        "pyclipper",
        "shapely",
        "omegaconf",
        "numpy",
    ):
        try:
            print(f"{dep:24} {importlib.metadata.version(dep)}")
        except Exception:
            print(f"{dep:24} MISSING")
    try:
        import rapidocr

        print("rapidocr", rapidocr.__file__)
    except Exception:
        traceback.print_exc()
    try:
        from core_pipeline.ocr import RapidOCRBackend

        backend = RapidOCRBackend(lang="ch", device="auto")
        backend._ensure_model()
        print(
            "RapidOCR init OK",
            backend.spec.label,
            backend.active_providers or backend.providers,
        )
    except Exception:
        traceback.print_exc()
    try:
        from backend.config import get_settings
        from core_pipeline.runtime import describe_runtime

        settings = get_settings()
        print(f"lama_model               {settings.resolved_lama_path}")
        print(
            f"lama_available           {settings.lama_model_available} "
            f"({settings.lama_model_source})"
        )
        print(f"rapidocr_models_dir      {settings.rapidocr_models_dir}")
        print(f"ort_runtime              {describe_runtime()}")
        try:
            import onnxruntime as ort

            print(
                f"ort_active               {ort.__version__} {ort.__file__} "
                f"{ort.get_available_providers()}"
            )
        except Exception:
            traceback.print_exc()
    except Exception:
        traceback.print_exc()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--doctor" in args:
        return _run_doctor()
    if "--backend" in args:
        return _run_backend()
    return _run_gui(show_window="--show" in args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
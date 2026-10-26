"""Shared ONNX Runtime execution-provider selection and session defaults.

O-Translator runs two very different ONNX workloads (LaMa inpainting and the
optional ONNX OCR adapter), so the provider order, thread caps, and fallback
behaviour live here instead of being duplicated in each backend.

Provider policy (highest priority first):

    CUDAExecutionProvider -> OpenVINOExecutionProvider ->
    DmlExecutionProvider -> CPUExecutionProvider

Only providers that the installed ``onnxruntime`` build actually exposes are
used. ``onnxruntime`` (CPU), ``onnxruntime-gpu``, ``onnxruntime-openvino`` and
``onnxruntime-directml`` all publish the same package name, so users install the
variant that matches their hardware and the rest of this module adapts.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CPU_PROVIDER = "CPUExecutionProvider"
PROVIDER_PRIORITY: tuple[str, ...] = (
    "CUDAExecutionProvider",
    "OpenVINOExecutionProvider",
    "DmlExecutionProvider",
    CPU_PROVIDER,
)

# Friendly aliases accepted by OTR_LAMA_DEVICE / OTR_ORT_PROVIDERS.
_DEVICE_ALIASES = {
    "directml": "dml",
    "gpu": "auto",
    "openvino": "openvino",
    "ov": "openvino",
    "cuda": "cuda",
    "cpu": "cpu",
    "auto": "auto",
}

_available_lock = threading.Lock()
_available_cache: list[str] | None = None

_gpu_activation_lock = threading.Lock()
_gpu_runtime_path: Path | None = None
_gpu_runtime_activated = False
_gpu_dll_handles: list[Any] = []


def _gpu_runtime_candidates() -> list[Path]:
    """Places an optional CUDA runtime add-on may be installed."""
    candidates: list[Path] = []
    explicit = os.environ.get("OTR_GPU_RUNTIME_DIR")
    if explicit:
        candidates.append(Path(explicit))
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / "gpu_runtime")
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "gpu_runtime")
    else:
        candidates.append(Path(__file__).resolve().parent.parent / "gpu_runtime")
    return candidates


def activate_gpu_runtime() -> Path | None:
    """Prefer an optional self-contained CUDA ONNX Runtime add-on.

    A portable CUDA build is large (CUDA/cuBLAS/cuDNN DLLs are several GB), so
    the base package stays CPU-only and an add-on zip can be extracted next to
    ``O-Translator.exe`` as ``gpu_runtime/``. When present it is placed before
    the bundled CPU ``onnxruntime`` on ``sys.path`` and its DLL directories are
    added to the Windows search path. If no add-on exists, the normal bundled
    CPU runtime is used unchanged.

    Must run before the first ``import onnxruntime``; this module calls it from
    ``available_providers``/``create_session`` and the entry points call it at
    process startup.
    """
    global _gpu_runtime_activated, _gpu_runtime_path
    if _gpu_runtime_activated:
        return _gpu_runtime_path
    if os.environ.get("OTR_DISABLE_GPU", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    with _gpu_activation_lock:
        if _gpu_runtime_activated:
            return _gpu_runtime_path
        for root in _gpu_runtime_candidates():
            package = root / "onnxruntime"
            if not package.is_dir():
                continue
            dll_dirs = [package / "capi"]
            nvidia_root = root / "nvidia"
            if nvidia_root.is_dir():
                dll_dirs.extend(
                    sorted(path for path in nvidia_root.glob("*/bin") if path.is_dir())
                )
            path_entries = [str(path) for path in dll_dirs if path.is_dir()]
            if path_entries:
                os.environ["PATH"] = os.pathsep.join(
                    [*path_entries, os.environ.get("PATH", "")]
                )
                for entry in path_entries:
                    with contextlib.suppress(AttributeError, OSError):
                        _gpu_dll_handles.append(os.add_dll_directory(entry))
            # Prepend so ``import onnxruntime`` picks the CUDA package rather
            # than the CPU copy collected into ``_internal``.
            if "onnxruntime" in sys.modules:
                logger.warning(
                    "onnxruntime was imported before the GPU add-on; restart the "
                    "process to activate %s",
                    root,
                )
            else:
                sys.path.insert(0, str(root))
            _gpu_runtime_path = root.resolve()
            _gpu_runtime_activated = True
            logger.info(
                "Activated external GPU ONNX Runtime add-on: %s", _gpu_runtime_path
            )
            return _gpu_runtime_path
    return None


def gpu_runtime_path() -> Path | None:
    return _gpu_runtime_path


def _normalize_device(device: str | None) -> str:
    value = (device or "auto").strip().lower()
    return _DEVICE_ALIASES.get(value, value)


def available_providers(*, refresh: bool = False) -> list[str]:
    """Return the execution providers compiled into the installed ORT build."""
    global _available_cache
    activate_gpu_runtime()
    with _available_lock:
        if _available_cache is not None and not refresh:
            return list(_available_cache)
        try:
            import onnxruntime as ort

            providers = [str(item) for item in ort.get_available_providers()]
        except Exception:  # pragma: no cover - onnxruntime optional / broken install
            providers = []
        _available_cache = providers
        return list(providers)


def parse_provider_list(value: str | None) -> list[str]:
    """Parse a comma/space separated provider list from configuration."""
    if not value:
        return []
    raw = value.replace(";", ",").replace(" ", ",")
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_providers(
    device: str = "auto",
    *,
    explicit: str | None = None,
    available: Sequence[str] | None = None,
) -> list[str]:
    """Resolve a human-facing device preference into an ORT provider list.

    The result always ends with ``CPUExecutionProvider`` so a partially
    unsupported graph can still finish, mirroring ORT's own fallback contract.
    """
    current = list(available) if available is not None else available_providers()
    if not current:
        return [CPU_PROVIDER]

    requested = parse_provider_list(explicit)
    if requested:
        chosen = [name for name in requested if name in current]
        if CPU_PROVIDER not in chosen and CPU_PROVIDER in current:
            chosen.append(CPU_PROVIDER)
        if chosen:
            return chosen

    normalized = _normalize_device(device)
    if normalized == "cpu":
        return [CPU_PROVIDER]
    if normalized == "cuda":
        preferred = ["CUDAExecutionProvider", CPU_PROVIDER]
    elif normalized == "openvino":
        preferred = ["OpenVINOExecutionProvider", CPU_PROVIDER]
    elif normalized in {"dml", "directml"}:
        preferred = ["DmlExecutionProvider", CPU_PROVIDER]
    else:
        preferred = list(PROVIDER_PRIORITY)

    chosen = [name for name in preferred if name in current]
    if CPU_PROVIDER in current and CPU_PROVIDER not in chosen:
        chosen.append(CPU_PROVIDER)
    return chosen or [CPU_PROVIDER]


def provider_options(
    providers: Sequence[str],
    *,
    openvino_device: str = "CPU",
    openvino_threads: int = 0,
    dml_device_id: int = 0,
    cuda_device_id: int = 0,
) -> list[str | tuple[str, dict[str, str]]]:
    """Attach per-provider options while preserving the fallback order."""
    entries: list[str | tuple[str, dict[str, str]]] = []
    for name in providers:
        if name == "OpenVINOExecutionProvider":
            options = {"device_type": openvino_device or "CPU"}
            if openvino_threads > 0:
                options["num_of_threads"] = str(openvino_threads)
            entries.append((name, options))
        elif name == "DmlExecutionProvider":
            entries.append((name, {"device_id": str(max(0, dml_device_id))}))
        elif name == "CUDAExecutionProvider":
            entries.append((name, {"device_id": str(max(0, cuda_device_id))}))
        else:
            entries.append(name)
    return entries


def session_options(
    *,
    threads: int = 0,
    inter_threads: int = 1,
    graph_optimization: str = "all",
    enable_mem_pattern: bool = True,
) -> Any:
    """Create SessionOptions with the settings shared by both workloads."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    level = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }.get((graph_optimization or "all").lower(), ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    options.graph_optimization_level = level
    options.intra_op_num_threads = max(0, int(threads))
    options.inter_op_num_threads = max(1, int(inter_threads))
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = bool(enable_mem_pattern)
    options.log_severity_level = 3
    return options


def create_session(
    model_path: str,
    *,
    device: str = "auto",
    explicit_providers: str | None = None,
    threads: int = 0,
    inter_threads: int = 1,
    graph_optimization: str = "all",
    openvino_device: str = "CPU",
    openvino_threads: int = 0,
    dml_device_id: int = 0,
    cuda_device_id: int = 0,
) -> tuple[Any, list[str]]:
    """Create an InferenceSession, falling back to CPU if a provider fails.

    Returns ``(session, registered_providers)``. A hardware provider can be
    present in the build but fail to initialize (missing driver, unsupported
    CPU feature); retrying with CPU-only keeps the feature available instead of
    turning a driver problem into a failed screenshot.
    """
    activate_gpu_runtime()
    import onnxruntime as ort

    available = available_providers()
    providers = resolve_providers(device, explicit=explicit_providers, available=available)
    options = session_options(
        threads=threads,
        inter_threads=inter_threads,
        graph_optimization=graph_optimization,
    )
    entries = provider_options(
        providers,
        openvino_device=openvino_device,
        openvino_threads=openvino_threads,
        dml_device_id=dml_device_id,
        cuda_device_id=cuda_device_id,
    )
    try:
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=entries)
        registered = list(session.get_providers())
        logger.info(
            "ONNX session %s providers=%s (requested=%s)",
            model_path,
            registered,
            providers,
        )
        return session, registered
    except Exception as exc:
        if providers == [CPU_PROVIDER]:
            raise
        logger.warning(
            "ONNX session with providers %s failed (%s); retrying CPU-only",
            providers,
            exc,
        )
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=[CPU_PROVIDER]
        )
        return session, list(session.get_providers())


def describe_runtime() -> dict[str, object]:
    """Safe summary for ``/api/health`` and startup logs."""
    available = available_providers()
    preferred = resolve_providers("auto", available=available)
    return {
        "providers_available": available,
        "providers_preferred": preferred,
        "hardware_acceleration": any(name != CPU_PROVIDER for name in preferred),
        "gpu_runtime": str(_gpu_runtime_path) if _gpu_runtime_path else None,
    }


def first_available(names: Iterable[str]) -> str | None:
    current = set(available_providers())
    for name in names:
        if name in current:
            return name
    return None


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}
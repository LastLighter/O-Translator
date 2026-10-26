from __future__ import annotations

import sys

import core_pipeline.runtime as runtime
from core_pipeline.runtime import (
    CPU_PROVIDER,
    describe_runtime,
    provider_options,
    resolve_providers,
)


def test_auto_provider_order_is_cuda_openvino_dml_cpu():
    available = [
        "CPUExecutionProvider",
        "DmlExecutionProvider",
        "OpenVINOExecutionProvider",
        "CUDAExecutionProvider",
    ]
    assert resolve_providers("auto", available=available) == [
        "CUDAExecutionProvider",
        "OpenVINOExecutionProvider",
        "DmlExecutionProvider",
        CPU_PROVIDER,
    ]


def test_cpu_device_never_selects_hardware_provider():
    available = ["CUDAExecutionProvider", CPU_PROVIDER]
    assert resolve_providers("cpu", available=available) == [CPU_PROVIDER]


def test_explicit_provider_list_keeps_cpu_fallback():
    available = ["DmlExecutionProvider", CPU_PROVIDER]
    assert resolve_providers(
        "auto", explicit="DmlExecutionProvider", available=available
    ) == ["DmlExecutionProvider", CPU_PROVIDER]


def test_missing_hardware_provider_falls_back_to_cpu():
    available = [CPU_PROVIDER]
    assert resolve_providers("cuda", available=available) == [CPU_PROVIDER]
    assert resolve_providers("openvino", available=available) == [CPU_PROVIDER]


def test_provider_options_are_serializable():
    entries = provider_options(
        ["OpenVINOExecutionProvider", "DmlExecutionProvider", CPU_PROVIDER],
        openvino_device="GPU",
        openvino_threads=4,
        dml_device_id=1,
    )
    assert entries[0] == (
        "OpenVINOExecutionProvider",
        {"device_type": "GPU", "num_of_threads": "4"},
    )
    assert entries[1] == ("DmlExecutionProvider", {"device_id": "1"})
    assert entries[2] == CPU_PROVIDER


def test_runtime_description_is_safe():
    summary = describe_runtime()
    assert "providers_available" in summary
    assert "providers_preferred" in summary
    assert isinstance(summary["hardware_acceleration"], bool)
    assert "gpu_runtime" in summary


def test_optional_gpu_addon_is_preferred_when_present(monkeypatch, tmp_path):
    addon = tmp_path / "gpu_runtime"
    (addon / "onnxruntime" / "capi").mkdir(parents=True)
    (addon / "nvidia" / "cudnn" / "bin").mkdir(parents=True)
    before_path = list(sys.path)
    monkeypatch.setenv("OTR_GPU_RUNTIME_DIR", str(addon))
    monkeypatch.setattr(runtime, "_gpu_runtime_activated", False)
    monkeypatch.setattr(runtime, "_gpu_runtime_path", None)
    # Other tests may have already imported the CPU runtime; activation refuses
    # to shadow an imported module, so simulate a fresh process here.
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    try:
        activated = runtime.activate_gpu_runtime()
        assert activated == addon.resolve()
        assert sys.path[0] == str(addon)
        assert runtime.gpu_runtime_path() == addon.resolve()
        assert runtime.activate_gpu_runtime() == addon.resolve()
    finally:
        sys.path[:] = before_path
        runtime._gpu_runtime_activated = False
        runtime._gpu_runtime_path = None


def test_gpu_addon_can_be_disabled(monkeypatch, tmp_path):
    addon = tmp_path / "gpu_runtime"
    (addon / "onnxruntime").mkdir(parents=True)
    monkeypatch.setenv("OTR_GPU_RUNTIME_DIR", str(addon))
    monkeypatch.setenv("OTR_DISABLE_GPU", "1")
    monkeypatch.setattr(runtime, "_gpu_runtime_activated", False)
    monkeypatch.setattr(runtime, "_gpu_runtime_path", None)
    try:
        assert runtime.activate_gpu_runtime() is None
    finally:
        runtime._gpu_runtime_activated = False
        runtime._gpu_runtime_path = None
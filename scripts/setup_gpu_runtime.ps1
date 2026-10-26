# Prepare the optional CUDA ONNX Runtime add-on used by O-Translator.
#
#   powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_runtime.ps1
#
# This creates .venv-gpu-cuda (isolated from the CPU package) with:
#   * onnxruntime-gpu 1.24.4 (CUDA 12.x / cuDNN 9.x build)
#   * NVIDIA cuBLAS/cuDNN/cuFFT/cuRAND runtime wheels
# and then assembles dist\O-Translator-gpu-runtime-cuda12.zip.
#
# The base O-Translator package remains CPU-only. Extract the add-on next to
# O-Translator.exe (gpu_runtime\) and CUDA is preferred automatically.

param(
    [string]$Venv = ".venv-gpu-cuda",
    [string]$Index = "https://pypi.tuna.tsinghua.edu.cn/simple",
    [string]$Python = "python",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$venvPython = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating $Venv (system site packages)..." -ForegroundColor Cyan
    & $Python -m venv --system-site-packages $Venv
}

Write-Host "Installing CUDA ONNX Runtime and NVIDIA runtime DLLs..." -ForegroundColor Cyan
& $venvPython -m pip install -i $Index --no-deps `
    onnxruntime-gpu==1.24.4 `
    nvidia-cuda-runtime-cu12==12.6.77 `
    nvidia-cublas-cu12==12.6.4.1 `
    nvidia-cudnn-cu12==9.5.1.17 `
    nvidia-cufft-cu12==11.3.3.83 `
    nvidia-curand-cu12==10.3.7.77
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the CUDA runtime packages."
}

if (-not $SkipBuild) {
    & $Python scripts\build_gpu_runtime.py
    if ($LASTEXITCODE -ne 0) {
        throw "GPU runtime packaging failed."
    }
}

Write-Host "Done." -ForegroundColor Green
Write-Host "Extract dist\O-Translator-gpu-runtime-cuda12.zip next to O-Translator.exe," -ForegroundColor Green
Write-Host "or set OTR_GPU_RUNTIME_DIR to the extracted gpu_runtime folder." -ForegroundColor Green

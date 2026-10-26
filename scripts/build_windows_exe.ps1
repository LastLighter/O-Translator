# Build the portable Windows desktop client with PyInstaller.
#
#   powershell -ExecutionPolicy Bypass -File scripts/build_windows_exe.ps1
#
# Output: dist\O-Translator\O-Translator.exe (double-click to run)
#
# Options:
#   -NoOcr      skip RapidOCR/ONNX Runtime (fast UI-only smoke build)
#   -NoPaddle   deprecated alias for -NoOcr
#   -SkipIcon   reuse build\icon.ico
#   -Clean      force a full PyInstaller cache rebuild (default reuses cache)
#   -DistPath   output directory (default "dist")
#   -ZipPath    portable zip target (default "dist\O-Translator-windows-x64.zip")

param(
    [switch]$NoOcr,
    [switch]$NoPaddle,
    [switch]$SkipIcon,
    [switch]$Clean,
    [string]$DistPath = "dist",
    [string]$ZipPath = "dist\O-Translator-windows-x64.zip"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not $SkipIcon) {
    Write-Host "Rendering application icon..." -ForegroundColor Cyan
    python scripts/make_icon.py
}

if ($NoOcr -or $NoPaddle) {
    $env:OTR_BUNDLE_OCR = "0"
    Write-Host "Building UI-only (OCR disabled)..." -ForegroundColor Yellow
} else {
    Remove-Item Env:OTR_BUNDLE_OCR -ErrorAction SilentlyContinue
    Remove-Item Env:OTR_BUNDLE_PADDLE -ErrorAction SilentlyContinue
    Write-Host "Building with RapidOCR (ONNX)..." -ForegroundColor Cyan
}

# Bundle the pre-folded LaMa graph so extraction is enough: no first-launch
# ~200 MB download, no 30 s dynamic-graph optimization.
$lamaCandidates = @(
    "models\lama\lama_fp32_sim.onnx",
    "models\lama\lama_fp32_fp16.onnx",
    "models\lama\lama_fp32.onnx"
)
$lamaBundle = $lamaCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $lamaBundle) {
    throw "Missing LaMa ONNX model. Run scripts/download_models.py --lama and scripts/optimize_lama.py first."
}
$env:OTR_LAMA_BUNDLE_MODEL = $lamaBundle
Write-Host "Bundling LaMa model: $lamaBundle" -ForegroundColor Cyan

$pyinstallerArgs = @(
    "--noconfirm",
    "--distpath", $DistPath,
    "--workpath", "build",
    "O-Translator.spec"
)
if ($Clean) {
    $pyinstallerArgs = @("--clean") + $pyinstallerArgs
}

$target = Join-Path $DistPath "O-Translator"
$gpuRuntime = Join-Path $target "gpu_runtime"
$gpuStash = Join-Path $DistPath ".O-Translator-gpu-runtime-stash"
$gpuPack = Join-Path "build" "gpu_runtime_pack\gpu_runtime"

# PyInstaller's COLLECT removes the whole onedir target, including the CUDA
# add-on copied in after an earlier build. Stash it first; otherwise a rebuild
# silently produces a CPU-only package that fails verify_dist.py.
if (Test-Path -LiteralPath $gpuRuntime) {
    if (Test-Path -LiteralPath $gpuStash) {
        Remove-Item -LiteralPath $gpuStash -Recurse -Force
    }
    Move-Item -LiteralPath $gpuRuntime -Destination $gpuStash
} elseif (-not (Test-Path -LiteralPath $gpuPack)) {
    throw "Missing gpu_runtime. Run scripts/build_gpu_runtime.py first; a complete release requires the CUDA runtime."
}

try {
    python -m PyInstaller @pyinstallerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
} finally {
    Remove-Item Env:OTR_LAMA_BUNDLE_MODEL -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    if (Test-Path -LiteralPath $gpuStash) {
        Move-Item -LiteralPath $gpuStash -Destination $gpuRuntime
    } elseif (Test-Path -LiteralPath $gpuPack) {
        New-Item -ItemType Directory -Force -Path $gpuRuntime | Out-Null
        robocopy $gpuPack $gpuRuntime /E /NFL /NDL /NJH /NJS /NP /R:1 /W:1 | Out-Null
        if ($LASTEXITCODE -ge 8) {
            throw "Failed to copy gpu_runtime (robocopy exit $LASTEXITCODE)."
        }
    }
}

$cudaProvider = Join-Path $gpuRuntime "onnxruntime\capi\onnxruntime_providers_cuda.dll"
if (-not (Test-Path -LiteralPath $cudaProvider)) {
    throw "The packaged build has no CUDA provider. Restore gpu_runtime and rebuild."
}

New-Item -ItemType Directory -Force -Path (Join-Path $target "data"), (Join-Path $target "models") | Out-Null
Copy-Item "packaging\README-Windows.txt" (Join-Path $target "使用说明.txt") -Force
Remove-Item (Join-Path $target "data\*") -Recurse -Force -ErrorAction SilentlyContinue
# The model is intentionally bundled under _internal\models\lama. Do not copy
# the repository's model directory into the writable exe folder as well.
Remove-Item (Join-Path $target "models\*") -Recurse -Force -ErrorAction SilentlyContinue

$bundledModel = Get-ChildItem (Join-Path $target "_internal\models\lama") -Filter *.onnx -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $bundledModel) {
    throw "The packaged build does not contain a bundled LaMa model."
}
Write-Host ("Bundled model: {0} ({1:N0} MB)" -f $bundledModel.Name, ($bundledModel.Length / 1MB)) -ForegroundColor Green

$ocrModels = @()
$ocrModels += Get-ChildItem (Join-Path $target "_internal\rapidocr\models") -Filter *.onnx -ErrorAction SilentlyContinue
$ocrModels += Get-ChildItem (Join-Path $target "_internal\models\rapidocr") -Filter *.onnx -ErrorAction SilentlyContinue
if ($ocrModels.Count -eq 0) {
    throw "The packaged build does not contain RapidOCR ONNX models."
}
Write-Host ("Bundled OCR models: {0} files, {1:N0} MB" -f $ocrModels.Count, (($ocrModels | Measure-Object Length -Sum).Sum / 1MB)) -ForegroundColor Green

python scripts/package_zip.py $target $ZipPath
if ($LASTEXITCODE -ne 0) {
    throw "Zip packaging failed with exit code $LASTEXITCODE"
}

$size = (Get-ChildItem $target -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ("Built {0} ({1:N0} MB)" -f (Join-Path $target "O-Translator.exe"), ($size / 1MB)) -ForegroundColor Green
# Smoke-test the packaged Windows client against a real job.
#
#   powershell -ExecutionPolicy Bypass -File scripts/smoke_test_exe.ps1
#
# Starts the bundled backend directly (no GUI), submits examples\demo_manga.png,
# waits for the pipeline, then prints the engines and translated text.

param(
    [string]$Root = "dist\O-Translator",
    [string]$Image = "examples\demo_manga.png",
    [int]$Port = 8765,
    [string]$GpuRuntimeDir = ""
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$exe = Join-Path $Root "O-Translator.exe"
if (-not (Test-Path $exe)) {
    throw "Missing $exe. Run scripts/build_windows_exe.ps1 first."
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use. Stop the running backend first."
}

$env:OTR_PORT = "$Port"
if ($GpuRuntimeDir) {
    $env:OTR_GPU_RUNTIME_DIR = (Resolve-Path $GpuRuntimeDir).Path
}
$logDir = Join-Path $Root "data"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$outLog = Join-Path $logDir "smoke-backend.out.log"
$errLog = Join-Path $logDir "smoke-backend.err.log"
$process = Start-Process -FilePath $exe -ArgumentList "--backend" -PassThru `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog
$base = "http://127.0.0.1:$Port"

try {
    $deadline = (Get-Date).AddSeconds(120)
    $health = $null
    while ((Get-Date) -lt $deadline) {
        try {
            $health = Invoke-RestMethod "$base/api/health" -TimeoutSec 3
            break
        } catch {
            Start-Sleep -Milliseconds 800
        }
    }
    if (-not $health) { throw "Backend did not become ready." }

    Write-Host ("OCR={0} inpaint={1} translation={2}" -f `
        $health.engines.ocr.active, $health.engines.inpainting.active, `
        $health.engines.translation.active) -ForegroundColor Cyan
    Write-Host ("OCR: version={0} profile={1} providers={2}" -f `
        $health.engines.ocr.version, $health.engines.ocr.profile, `
        ($health.engines.ocr.providers -join ", ")) -ForegroundColor Cyan
    Write-Host ("LaMa: present={0} source={1} precision={2}" -f `
        $health.engines.inpainting.lama_weights_present, `
        $health.engines.inpainting.lama_source, `
        $health.engines.inpainting.lama_precision) -ForegroundColor Cyan
    Write-Host ("providers: " + ($health.engines.inpainting.providers -join ", ")) -ForegroundColor Cyan
    Write-Host ("runtime: " + ($health.runtime | ConvertTo-Json -Compress)) -ForegroundColor Cyan
    if (-not $health.engines.ocr.rapidocr_installed) {
        throw "Packaged build cannot import RapidOCR."
    }
    if ($health.engines.ocr.active -ne "rapidocr") {
        throw "Packaged build is not using the RapidOCR engine: $($health.engines.ocr.active)"
    }
    if (-not $health.engines.inpainting.lama_weights_present) {
        throw "Packaged build cannot find the bundled LaMa model."
    }
    if ($null -ne $health.engines.inpainting.download) {
        throw "Packaged build unexpectedly started a LaMa download."
    }
    if ($health.startup_issues.Count -gt 0) {
        Write-Host ("startup issues: " + ($health.startup_issues -join "; ")) -ForegroundColor Yellow
    }

    $response = curl.exe -s -X POST "$base/api/jobs" `
        -F "file=@$Image" -F "source_lang=ja" -F "target_lang=zh" `
        -F "mode=manga" -F "reading_order=rtl"
    $job = $response | ConvertFrom-Json
    if (-not $job.id) { throw "Job submission failed: $response" }
    Write-Host "job $($job.id)" -ForegroundColor Cyan

    $deadline = (Get-Date).AddMinutes(6)
    $state = $null
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        $state = Invoke-RestMethod "$base/api/jobs/$($job.id)" -TimeoutSec 5
        if ($state.status -in @("completed", "failed", "cancelled")) { break }
    }
    if ($state.status -ne "completed") {
        if (Test-Path $errLog) { Get-Content $errLog -Tail 40 | Write-Host }
        throw "Job ended as $($state.status): $($state.error | ConvertTo-Json -Compress)"
    }

    $result = $state.result
    Write-Host ("engines: " + ($result.engines | ConvertTo-Json -Compress)) -ForegroundColor Green
    Write-Host ("translation: " + ($result.translation | ConvertTo-Json -Compress)) -ForegroundColor Green
    Write-Host ("timings: " + ($result.timings_ms | ConvertTo-Json -Compress)) -ForegroundColor Green
    if ($state.preview) {
        Write-Host ("preview: " + $state.preview.image_url) -ForegroundColor Green
    }
    Write-Host ("regions: {0}, blocks: {1}" -f $result.regions.Count, $result.blocks.Count)
    $result.regions | Select-Object -First 5 | ForEach-Object {
        Write-Host ("  [{0:N3}] {1} -> {2}" -f $_.confidence, $_.text, $_.translation)
    }
    Write-Host "OK" -ForegroundColor Green
} finally {
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
    }
    Remove-Item Env:OTR_PORT -ErrorAction SilentlyContinue
    Remove-Item Env:OTR_GPU_RUNTIME_DIR -ErrorAction SilentlyContinue
}
# LaMa model weights

O-Translator does not commit model weights to source control. Two inpainting
paths are available:

1. **LaMa (recommended for artwork)** — an ONNX export of *Resolution-robust
   Large Mask Inpainting with Fourier Convolutions* (WACV 2022).
2. **OpenCV Telea / solid fill** — always available, fast (<15–30 ms), and
   visually indistinguishable for UI boxes, game chat boxes, and flat manga
   speech bubbles. `AutoInpainter` uses it automatically for low-variance
   regions even when LaMa is installed.

## Packaged Windows build (no download)

`scripts/build_windows_exe.ps1` bundles the folded graph
`models/lama/lama_fp32_sim.onnx` under `_internal/models/lama/`. The backend
searches both the writable exe folder and that bundled read-only directory, so
**extract-and-run never starts a LaMa download**. `/api/health` reports:

```json
{
  "inpainting": {
    "lama_weights_present": true,
    "lama_source": "bundled",
    "lama_path": "...\\_internal\\models\\lama\\lama_fp32_sim.onnx",
    "download": null
  }
}
```

The optional precision variants (`lama_fp32_fp16.onnx`, `lama_fp32_int8.onnx`)
are *not* bundled: on CPU they were both slower and numerically worse for
LaMa's FFC/FFT blocks. They remain opt-in for GPU providers.

## Download from source checkouts

The backend downloads the weights automatically in the background on first start
when `OTR_LAMA_AUTO_DOWNLOAD` is enabled (the default) **and no optimized graph
is already present**. This is the same downloader used by the CLI, with proxy
and mirror support:

```powershell
python scripts/download_models.py --lama --optimize
python scripts/download_models.py --check
# Browser-only proxy (Clash Verge/Meta mixed port):
python scripts/download_models.py --lama --proxy http://127.0.0.1:7897
```

Default destination:

```
models/lama/lama_fp32.onnx
```

The downloader tries `hf-mirror.com`, then `huggingface.co`. If both are
unreachable, download the model manually from
[Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) and place
`lama_fp32.onnx` at the path above. You can also pass a custom mirror:

```powershell
python scripts/download_models.py --lama --url https://example.com/lama_fp32.onnx
```

Or configure the server once in `.env`:

```dotenv
OTR_PROXY=http://127.0.0.1:7897
OTR_HF_ENDPOINT=https://hf-mirror.com
```

`/api/health` exposes the download status, bytes transferred, source URL, the
resolved model path/source, and the active ORT provider order.

The expected file is roughly 198 MB and exposes two ONNX inputs (`image` and
`mask`). `--check` prints the actual input/output shapes.

## Faster startup (recommended on CPU)

The Carve export has a dynamic 17,480-node graph with the FFT unrolled into
primitive ops. ONNX Runtime spends 25–35 s optimizing it when the session is
created. Fixing the already-fixed spatial dimensions (512×512) lets `onnxsim`
constant-fold the shape machinery:

```powershell
python -m pip install onnxsim
python scripts/optimize_lama.py --verify
```

This writes `lama_fp32_sim.onnx` next to the original (3,028 nodes, ~196 MB).
The inpainter automatically prefers it; loading drops to 1.6–2.3 s with
identical output (`--verify` prints the numeric diff). The original file is
still used if the optimized one is missing or cannot be loaded.

### Precision reduction (opt-in)

`scripts/optimize_lama.py --precision fp16|int8 --verify` writes
`lama_fp32_fp16.onnx` / `lama_fp32_int8.onnx`. They are selected only when
`OTR_LAMA_PRECISION` asks for that family. Measured on this project's Intel CPU:

| Variant | Size | CPU inference | Max output diff vs FP32 |
|---|---:|---:|---:|
| FP32 folded | ~196 MB | ~3.0–3.8 s | 0.0005 (graph folding only) |
| FP16 | ~99 MB | ~5.6–6.2 s | ~204/255 (unusable on CPU) |
| Dynamic INT8 | ~55 MB | ~5.4–41 s | ~163/255 (unusable) |

FP16/INT8 are therefore **not** the CPU default and are not bundled. They may
be useful with a real CUDA/DirectML/OpenVINO-GPU session where FP16 kernels are
native; always run `--verify` on your hardware before enabling them.

### Execution providers

`core_pipeline/runtime.py` applies one provider order to LaMa and the optional
ONNX OCR adapter:

```
CUDAExecutionProvider -> OpenVINOExecutionProvider ->
DmlExecutionProvider -> CPUExecutionProvider
```

Only providers compiled into the installed `onnxruntime` variant are used, and
session creation retries CPU-only if a driver/provider fails. Install
`onnxruntime-gpu`, `onnxruntime-openvino`, or `onnxruntime-directml` (one at a
time; they share the same import name) and select it with `OTR_LAMA_DEVICE` or
`OTR_LAMA_PROVIDERS`.

For the portable Windows build, the base zip is CPU-only. An optional
self-contained CUDA add-on can be extracted next to `O-Translator.exe`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_runtime.ps1
# dist\O-Translator-gpu-runtime-cuda12.zip -> extract gpu_runtime\ next to the exe
```

`core_pipeline/runtime.py` activates it before `import onnxruntime`, so
`CUDAExecutionProvider` is preferred automatically. Verified on an RTX 5060
Laptop: warm LaMa 512×512 inference drops from ~2.1 s (CPU) to ~0.21–0.30 s,
with max output difference 0.47/255 and no pixel >1.

`OTR_LAMA_THREADS` caps ONNX Runtime's intra-op threads (default `0` = auto,
capped at 4). The FFC/FFT graph stops scaling after a few threads, and using
every logical core is slower on a busy hybrid CPU.

## Model contract

The loader supports the common export variants:

| Aspect | Supported |
|---|---|
| Inputs | separate `image` + `mask`; dynamic or fixed H/W |
| Image range | `[0,1]`, `[-1,1]`, or ImageNet normalization |
| Mask | single channel binary `{0,1}` (repeated automatically if needed) |
| Output | `[0,1]` or `[0,255]` float; auto-detected |

Environment overrides:

```dotenv
OTR_INPAINT_ENGINE=auto          # auto | lama | opencv
OTR_LAMA_MODEL_PATH=models/lama/lama_fp32.onnx
OTR_LAMA_PRECISION=auto          # auto | fp32 | fp16 | int8
OTR_LAMA_DEVICE=auto             # auto | cpu | cuda | openvino | dml
OTR_LAMA_PROVIDERS=              # explicit ORT order; empty = device policy
OTR_LAMA_OPENVINO_DEVICE=CPU     # CPU | GPU | AUTO
OTR_LAMA_DML_DEVICE_ID=0
OTR_LAMA_NORMALIZATION=zero_one  # zero_one | minus_one_one | imagenet
OTR_LAMA_AUTO_DOWNLOAD=true
OTR_LAMA_THREADS=0               # 0 = auto (max 4)
OTR_INPAINT_MAX_SIDE=1024
OTR_INPAINT_FAST_STDDEV=24       # below this, skip LaMa (fast local fill)
OTR_INPAINT_SOLID_STDDEV=3       # below this, plain median-colour fill
OTR_INPAINT_ROI_PADDING=16
OTR_INPAINT_MAX_ROI_PATCHES=1
OTR_PROGRESSIVE_PREVIEW=true
```

The Carve export is a **fixed 512×512** graph, so `OTR_INPAINT_MAX_SIDE` does
not change its inference cost: every ROI is resized to 512×512 and the result
is resized back and blended only through the text mask. The setting only
matters for dynamic-shape exports. Adaptive ROI cropping is a latency
optimization, not a way to make a 512×512 FFC network cheap on CPU: one union
ROI is ~3–4 s on the reference laptop CPU, while texture-free UI screenshots
skip LaMa entirely.

## Font

The renderer auto-detects Microsoft YaHei / SimHei on Windows and Noto Sans CJK
on Linux. For a self-contained deployment, place a licensed CJK TTF/OTF/TTC in
`models/fonts/` or point `OTR_FONT_PATH` at it. Font files are gitignored.

## Licenses

- LaMa reference implementation: Apache-2.0 (advimman/lama).
- The Carve ONNX export is published under Apache-2.0 on Hugging Face.
- PaddleOCR and PaddlePaddle follow their own Apache-2.0 licenses.
- Microsoft YaHei and other system fonts are **not** redistributable with this
  project; use a properly licensed font for commercial packaging.
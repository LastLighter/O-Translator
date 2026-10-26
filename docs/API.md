# HTTP API

Base URL: `http://127.0.0.1:8765`

All errors use the same envelope:

```json
{
  "code": "invalid_image",
  "message": "Unsupported or corrupted image file.",
  "details": {}
}
```

## `GET /api/health`

Returns safe runtime capabilities. It never includes the API key.

```json
{
  "status": "ok",
  "app": "O-Translator",
  "version": "0.1.0",
  "engines": {
    "ocr": {
      "active": "paddle",
      "paddle_installed": true,
      "tesseract_wrapper_installed": false,
      "configured": "auto",
      "paddle_lang": "ch",
      "paddle_profile": "auto",
      "paddle_det_limit_side_len": 1280
    },
    "inpainting": {
      "active": "lama-onnx",
      "lama_weights_present": true,
      "lama_path": "D:\\...\\models\\lama\\lama_fp32_sim.onnx",
      "lama_source": "bundled",
      "lama_precision": "auto",
      "lama_device": "auto",
      "providers": ["CPUExecutionProvider"],
      "fast_stddev": 24.0,
      "roi_padding": 16,
      "progressive_preview": true,
      "onnxruntime_installed": true,
      "opencv_fallback": true,
      "download": null
    },
    "translation": {
      "active": "deepseek",
      "configured_provider": "deepseek",
      "deepseek_configured": true,
      "model": "deepseek-chat"
    }
  },
  "runtime": {
    "providers_available": ["CPUExecutionProvider"],
    "providers_preferred": ["CPUExecutionProvider"],
    "hardware_acceleration": false,
    "gpu_runtime": null
  },
  "queue": {"workers": 2, "queued": 0, "active": 0},
  "limits": {"max_upload_mb": 20, "max_image_side": 2400},
  "startup_issues": []
}
```

When the optional CUDA add-on is installed, `runtime.providers_available`
includes `CUDAExecutionProvider`, `providers_preferred` starts with it,
`hardware_acceleration` becomes `true`, and `gpu_runtime` is the add-on path.

## `GET /api/languages`

Returns the language options used by the web UI.

## `POST /api/jobs`

`multipart/form-data`. Returns HTTP `202`.

| Field | Type | Default | Notes |
|---|---|---|---|
| `file` | file | required | PNG/JPEG/WEBP/BMP, max `OTR_MAX_UPLOAD_MB` |
| `source_lang` | string | `auto` | `auto`, `zh`, `en`, `ja`, `ko`, ... Also selects the OCR model (`ja`/`manga` use the Japanese profile) |
| `target_lang` | string | `zh` | target language |
| `mode` | string | `auto` | `auto`, `manga`, `game`, `novel`, `ui` |
| `reading_order` | string | `auto` | `auto`, `ltr`, `rtl` |
| `dilate_px` | int | server default | 3–7 (elliptical kernel diameter) |
| `font_scale` | float | `1.0` | 0.6–1.6 |
| `translate` | bool | `true` | false = OCR + inpaint + typeset source |

```bash
curl -X POST http://127.0.0.1:8765/api/jobs \
  -F "file=@examples/demo_game_ui.png" \
  -F "source_lang=zh" \
  -F "target_lang=en" \
  -F "mode=game" \
  -F "reading_order=ltr" \
  -F "dilate_px=5" \
  -F "font_scale=1.0"
```

Response:

```json
{
  "id": "20261018-023500-a1b2c3",
  "status": "queued",
  "stage": "queued",
  "progress": 0,
  "message": "Queued",
  "filename": "demo_game_ui.png",
  "options": {"source_lang": "zh", "target_lang": "en", "mode": "game"},
  "created_at": "2026-10-17T18:35:00Z",
  "updated_at": "2026-10-17T18:35:00Z",
  "warnings": [],
  "error": null,
  "result": null
}
```

## `GET /api/jobs/{job_id}`

Poll job state. Terminal statuses are `completed`, `failed`, and `cancelled`.
Completed jobs include:

```json
{
  "status": "completed",
  "progress": 100,
  "warnings": [],
  "result": {
    "job_id": "20261018-023500-a1b2c3",
    "image_url": "/api/jobs/.../files/result.png",
    "original_url": "/api/jobs/.../files/original.png",
    "clean_url": "/api/jobs/.../files/clean.png",
    "mask_url": "/api/jobs/.../files/mask.png",
    "width": 1400,
    "height": 900,
    "empty": false,
    "engines": {
      "ocr": "paddle",
      "inpainting": "hybrid-lama",
      "inpaint_strategy": "hybrid-roi"
    },
    "translation": {
      "provider": "deepseek",
      "model": "deepseek-chat",
      "degraded": false,
      "usage": {"prompt_tokens": 900, "completion_tokens": 300, "total_tokens": 1200}
    },
    "regions": [
      {
        "id": 0,
        "text": "旅人，你终于来了。",
        "translation": "Traveler, you have finally arrived.",
        "confidence": 0.98,
        "polygon": [[271, 764], [521, 764], [521, 803], [271, 803]],
        "bbox": [271, 764, 250, 39],
        "direction": "horizontal",
        "role": "text"
      }
    ],
    "blocks": [
      {
        "id": 0,
        "source_text": "旅人，你终于来了。",
        "translated_text": "Traveler, you have finally arrived.",
        "bbox": [271, 764, 250, 39],
        "direction": "horizontal",
        "layout": "horizontal",
        "font_size": 18,
        "line_height": 21,
        "lines": ["Traveler, you have", "finally arrived."],
        "text_color": [242, 239, 228],
        "background_color": [9, 13, 18],
        "overflow": false,
        "fit_ratio": 1.0
      }
    ],
    "timings_ms": {"ocr_ms": 480, "inpaint_ms": 820, "translation_ms": 1600, "render_ms": 65, "total_ms": 3050}
  }
}
```

`blocks` is intentionally rich enough for a client to render its own overlay
instead of using `result.png`.

Regions with `"role": "metadata"` (handles, timestamps, social headers) or
`"role": "keep"` (language-neutral counters such as `18` or `8.2万`) are still
listed so clients can show the line, but they have no `blocks` entry: the
pipeline never masks or redraws them, so their original pixels — avatars,
verified badges, action icons — stay intact.

While the job is still `running`, the same response can include a pass-1
preview:

```json
{
  "status": "running",
  "stage": "preview",
  "progress": 82,
  "preview": {
    "image_url": "/api/jobs/.../files/preview.png",
    "width": 1400,
    "height": 900,
    "preview": true,
    "blocks": [],
    "regions": [],
    "timings_ms": {"ocr_ms": 480, "inpaint_fast_ms": 34, "preview_ready_ms": 1675}
  }
}
```

The preview is the translation rendered over the fast (Telea/median) background.
Clients should display it immediately and then replace it with `result.png` when
the job reaches `completed`. `preview` remains in the terminal response for
before/after inspection.

## `GET /api/jobs/{job_id}/events`

Server-Sent Events stream. Events: `progress` (on each state change), `done`
(terminal), `error`. A keep-alive comment is emitted every ~10 seconds.

```javascript
const source = new EventSource(`/api/jobs/${id}/events`);
source.addEventListener("progress", (event) => console.log(JSON.parse(event.data)));
source.addEventListener("done", () => source.close());
```

## `GET /api/jobs/{job_id}/files/{filename}`

Serves one of: `result.png`, `original.png`, `clean.png`, `mask.png`,
`preview.png`, `metadata.json`. Other names return 404.

## `GET /api/jobs/{job_id}/download`

Same as `result.png` with `Content-Disposition: attachment` and a friendly
filename.

## Status codes

| Code | Meaning |
|---|---|
| 202 | Job accepted |
| 400 | Empty/invalid image |
| 404 | Unknown job/artifact |
| 413 | Upload exceeds limit |
| 422 | Field validation failed |
| 502 | DeepSeek upstream failed (job-level error) |
| 503 | OCR unavailable or queue full |
| 504 | DeepSeek timeout (job-level error) |

## OpenAPI

Interactive docs: <http://127.0.0.1:8765/api/docs>  
Schema: <http://127.0.0.1:8765/api/openapi.json>
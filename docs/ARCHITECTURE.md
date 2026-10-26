# Architecture

## Goals

O-Translator is optimized for interactive desktop use: a screenshot should
become a translated image in seconds, with predictable degradation when a model
or credential is unavailable. The architecture therefore separates heavyweight
inference from the API event loop and makes every stage replaceable.

## Module map

| Module | Responsibility | Important design choices |
|---|---|---|
| `backend/main.py` | App factory, lifespan, engine wiring | Lazy OCR loading; background LaMa download; model failures become startup warnings instead of fatal startup errors |
| `backend/api.py` | HTTP routes, multipart validation, SSE | Upload cap, image probe before queueing, structured errors, engine/download status |
| `backend/jobs.py` | Async queue + worker pool + state machine | Queue capacity, monotonic progress, per-job error payloads |
| `backend/artifacts.py` | Atomic artifact writes, public result shaping | Job-id path guard, whitelist of served files, no absolute paths leaked |
| `core_pipeline/pipeline.py` | End-to-end orchestration | Fast pass + translation overlap; LaMa ROI continues while pass-1 preview is emitted |
| `core_pipeline/ocr.py` | OCR adapters and reading order | PaddleOCR 3.x modules + classic pipeline, optional ONNX/RapidOCR adapter with the shared provider policy, native-size detection with upscaled recognition crops, zoomed caption/title retry, Japanese noise repair, mixed RTL/LTR reading order, Tesseract fallback |
| `core_pipeline/mask.py` | Polygon rasterization and dilation | Elliptical kernel, optional box expansion, merged-caption ribbon expansion |
| `core_pipeline/inpainting.py` | Adaptive fast fill + ROI LaMa + OpenCV Telea | Per-region background variance, local Telea/median fill, padded ROI crops, alpha/Poisson blend, lazy LaMa activation |
| `core_pipeline/runtime.py` | ONNX Runtime provider policy | `CUDA → OpenVINO → DML → CPU`, provider options, CPU retry, health summary |
| `core_pipeline/model_store.py` | Optional weight downloads | Progress state, proxy/HF-mirror support, atomic writes, `/api/health` reporting |
| `core_pipeline/translation.py` | DeepSeek batch translation | OCR-aware prompting, id reconciliation, retry/backoff, JSON extraction, script normalization, source fallback |
| `core_pipeline/typesetting.py` | Font fit, wrapping, colors, drawing | Pixel measuring, CJK/Latin line breaking, top-anchored vertical columns, display/bold faces, outlined lettering, contrast repair |
| `client/main.py` | Three-mode desktop window, tray, orchestration | Manual/area/monitor modes, async API workers, HUD targeting |
| `client/ui/overlay_snipper.py` | Per-screen snipping panes | One pane per QScreen avoids mixed-DPI scaling bugs; highlights the detected window/control and captures it on click/Space |
| `client/ui/floating_window.py` | Always-on-top translation HUD | Exact overlay geometry, direct image dragging, blue outer glow, per-source click-through, native hit test |
| `client/services/screen_capture.py` | QScreen ↔ mss mapping | Negative origins, measured scale factors, physical/logical conversion, capture |
| `client/services/ui_target.py` | Target geometry and selection model | Clipping, dedupe, area ordering, screen fallback, wheel selection stack |
| `client/services/win32_targets.py` | Fast Win32 z-order/child probe | Skips the overlay's own HWNDs, DWM frame bounds, HWND seed for UIA |
| `client/services/uia_targets.py` | Raw-`ctypes` UI Automation client | Control View descent from `ElementFromHandle`; no `comtypes` dependency |
| `client/services/target_detector.py` | Background detection pipeline | One-slot mailbox, 60 Hz coalescing, Win32→UIA staged signals, cache, stale-serial rejection |
| `client/services/region_monitor.py` | Fixed-region change monitor | QThread, grayscale diff, debounce, cooldown, exclusions |
| `client/services/hotkey.py` | Global hotkey | Windows RegisterHotKey + Qt native event filter |
| `client/web/` | Browser workbench | No framework/build step; polling + before/after comparator |

## Desktop capture and monitoring

The desktop client is intentionally separate from the server process:

```
GlobalHotkey / tray
        │
        ▼
OverlaySnipper (one pane per screen)
        │  logical selection + ScreenTarget
        ▼
CaptureRegion (physical pixels + logical HUD rect)
        │
        ├── one-shot capture ──► save PNG ──► ApiClient.create_job()
        │
        └── RegionMonitor QThread
                 ├─ mss grab(physical rect)
                 ├─ grayscale + downscale (≤480 px)
                 ├─ difference vs baseline and previous frame
                 ├─ pending → static for debounce → cooldown check
                 └─ trigger(frame) ──► save PNG ──► /api/jobs
```

`ScreenTarget` pairs each `QScreen` with an mss monitor by physical size and
origin. `CaptureRegion` keeps both coordinate systems because mss needs physical
pixels while Qt window placement needs logical pixels. A separate overlay pane
per screen keeps the snipper itself DPI-correct on mixed-scale desktops.

While the overlay is open, `TargetDetector` also maps the logical pointer to
physical pixels and runs a two-stage background probe: Win32 z-order/child
rectangles first, then UI Automation Control View elements seeded from the
underlying HWND. The overlay paints the innermost rectangle, the wheel changes
the selected layer, and a click/Space maps the chosen physical rect back through
`ScreenTarget.physical_rect_to_logical()` before the normal capture path. A
detached daemon thread and serial-based stale-result rejection keep mouse
movement responsive even when a UIA provider is slow. Details and failure modes
are in [DESKTOP.md](DESKTOP.md#automatic-windowcontrol-picking).

The monitor only performs cheap grayscale diffs. RGB conversion and QImage
creation happen on trigger. Translation jobs suspend the capture loop; the
floating HUD and main window rectangles are excluded from the diff so the
translated overlay cannot recursively trigger itself.


## Pipeline data flow

```
bytes
  │ decode_image()  (thread)
  ▼
PIL RGB ──► OCRBackend.read(image, source_lang=..., mode=...)  (thread, lock)
  │                │
  │                ├─ detect line polygons
  │                ├─ crop / rotate vertical columns
  │                ├─ rec_text + rec_score
  │                └─ TextRegion[]
  ▼
order_regions()  (row/column grouping, RTL for manga)
  │
  ▼
build_mask()  (prose polygons only; metadata/keep skipped; cv2.dilate)
  │
  ├────────────────────────────┐
  ▼                            ▼
fast_pass()                 Translator.translate()
  local Telea/median fill     async HTTP
  │                            │
  │  ┌── pass 1 preview ◄──────┤
  │  │   Renderer.render(fast) │
  ▼  ▼                         ▼
ROI LaMa (textured only)   id → translation
  padded crop → 512×512
  alpha/seamless blend
  │                            │
  ▼                            ▼
clean plate                translations
  └──────────────┬─────────────┘
                 ▼
          Renderer.render()
          (thread; font fit, colors, wrapping)
                 │
                 ▼
          PipelineResult
                 │
                 ▼
          ArtifactStore.save_result()
```

`TranslationPipeline.run(..., on_preview=...)` awaits translation first, renders
pass 1 on the fast canvas, and awaits the LaMa task afterwards. The backend
persists `preview.png` and publishes it on the job record while the job is still
`running`; the desktop client fetches it and cross-fades to `result.png` when the
final render arrives.

### Stage timing

`PipelineResult.timings_ms` records decode, OCR, mask, inpainting, translation,
render, and total. OCR models are initialized by a background warm-up thread at
server startup, so the first job normally starts with warm models; a job that
arrives before the warm-up finishes still pays part of that cost. In `auto`
language mode the page is recognized once and refined in place with the same
weights; only languages with different model signatures trigger a second full
pass. The renderer converts the source page to an RGB array once per render and
uses an additive-advance wrapper for CJK, which keeps typesetting around
10–35 ms on text-heavy pages. `LanguageAwareOCRBackend` keeps an 8-entry exact
page cache, so re-submitting an unchanged screenshot skips OCR (~13 ms instead
of ~1 s). See [PERFORMANCE.md](PERFORMANCE.md) for numbers and
`scripts/benchmark_pipeline.py` for the reproducible benchmark.

## Concurrency model

- FastAPI owns one asyncio event loop.
- `JobManager` starts `OTR_QUEUE_WORKERS` long-lived asyncio workers.
- Each worker runs one pipeline at a time, bounding CPU-heavy OCR/inpainting.
- Inside a pipeline:
  - image decode, OCR, mask generation, LaMa/OpenCV, and rendering run in
    `asyncio.to_thread`;
  - DeepSeek translation runs on the event loop using async `httpx`;
  - inpainting and translation are separate tasks and overlap;
  - translation chunks are limited by an `asyncio.Semaphore`.
- Progress callbacks only mutate the in-memory job record on the event loop.
- The desktop client performs HTTP calls in `QThreadPool` workers and applies
  results through Qt signals on the UI thread.

This gives two levels of concurrency without oversubscribing OCR:

```
2 queue workers × (1 OCR thread + 1 inpaint thread + N HTTP chunks)
```

## Job state machine

```
queued → running → completed
                 ├→ failed
                 └→ cancelled (shutdown)
```

Stages emitted to clients:

`decoding → ocr → masking → processing → preview → inpainting/translation → rendering → saving → complete`

Progress uses `max(previous, new)` to avoid visual jumps when the two parallel
stages report out of order.

## Error strategy

| Failure | Behavior |
|---|---|
| Invalid/corrupt/oversized image | HTTP 400/413 before queueing |
| OCR unavailable at boot | Server still starts; `/api/health` reports it; jobs fail with a clear error |
| OCR returns zero regions | Job succeeds with `empty: true`, original image, and a warning |
| LaMa missing | Bundled graph is used directly in packaged builds; otherwise a background download starts and fast local fill / Telea covers the gap |
| LaMa slower than preview | Pass-1 preview stays visible; pass-2 swaps in the refined result when the ROI inference finishes |
| Provider initialization fails | `runtime.create_session` retries CPU-only instead of failing the job |
| LaMa fails | Auto mode logs a warning and uses OpenCV Telea |
| LaMa requested at boot but unavailable | Startup warning; download continues in the background and LaMa activates lazily |
| DeepSeek timeout/HTTP error | Retries with exponential backoff; then source-text fallback if enabled |
| DeepSeek returns invalid JSON | Retried; missing ids fall back to source per line |
| DeepSeek rejects key | Non-retried 401/403 with `translation_auth_failed` |
| Renderer text overflow | Block is flagged `overflow: true`; job still completes |
| Unexpected pipeline exception | Job marked `failed` with `internal_error`; traceback logged |

Secrets are never included in health responses. The only URL exposed to clients
is the local artifact path.

## Mask and inpainting details

### Mask

1. Rasterize each OCR polygon with `cv2.fillPoly` at 255. `metadata` rows
   (handles, timestamps, social headers) and `keep` rows (language-neutral
   counters such as `18` or `8.2万`) are skipped entirely: their original
   pixels — avatars, verified badges, repost/like icons — must survive.
2. Optionally fill the expanded bounding rectangle (for noisy backgrounds).
3. Dilate with an ellipse of size `dilate_px × dilate_px` (constrained to
   3–7 px; default 7).
4. Save as a grayscale preview and use the binary array for inpainting.

Dilation handles anti-aliased glyph edges and JPEG ringing. The 7 px maximum is
usually enough for stylized lettering without erasing too much artwork.

### Adaptive fast inpainting

`AutoInpainter.fast_pass()` splits the mask into connected components, merges
nearby glyph fragments, and samples a ring of pixels just outside each component.
The per-channel standard deviation of that ring is the "is this background
simple?" signal:

| σ (mean channel stddev) | Action |
|---|---|
| `< OTR_INPAINT_SOLID_STDDEV` (3) | median-colour fill (fastest) |
| `< OTR_INPAINT_FAST_STDDEV` (24) | local `cv2.inpaint(..., INPAINT_TELEA)` |
| otherwise | padded ROI crop sent to LaMa, blended back |

The fast canvas is also the pass-1 preview background. Complex regions are
solid-filled for the preview and refined by LaMa afterwards. If LaMa is missing
or fails, the same regions are re-run through local Telea so quality degrades
gracefully.

### LaMa ROI preprocessing

- The bundled Carve export is a fixed 512×512 graph; the loader reads that shape
  from the session. `OTR_INPAINT_MAX_SIDE` only applies to dynamic-shape exports
  and is rounded down to a multiple of 8.
- Complex plans are clustered into at most `OTR_INPAINT_MAX_ROI_PATCHES` ROIs
  (default 1). Each ROI is `bbox + OTR_INPAINT_ROI_PADDING` px, resized to the
  model input, and blended back only through the complex-region mask.
- Resize RGB with bilinear interpolation, mask with nearest neighbor.
- Normalize image as `[0,1]`, `[-1,1]`, or ImageNet based on configuration.
- Feed `image` and `mask` by input name; repeat the mask channel if the graph
  expects 3 channels.
- Detect output range (`[0,1]`, `[-1,1]`, or `[0,255]`), clip, resize back.
- Blend through a 3×3-feathered mask (or `cv2.seamlessClone` when
  `OTR_INPAINT_BLEND=seamless`).

The dynamic export spends 25–35 s in ONNX Runtime graph optimization because the
FFT is unrolled into ~17k nodes. `scripts/optimize_lama.py` folds the fixed-shape
graph with `onnxsim` to ~3k nodes (same output); `LamaOnnxInpainter` prefers that
`*_sim.onnx` sibling. ORT inference is capped at four intra-op threads because
the FFT/Einsum subgraphs stop scaling after a few threads and otherwise starve
the OCR thread.

### Execution providers

`core_pipeline/runtime.py` resolves the installed ORT build's providers in this
order: `CUDAExecutionProvider`, `OpenVINOExecutionProvider`,
`DmlExecutionProvider`, `CPUExecutionProvider`. It attaches OpenVINO
`device_type`/thread options and DirectML/CUDA device ids, then retries CPU-only
if a provider fails to initialize. The same helper is used by LaMa and the
optional ONNX OCR adapter; `/api/health.runtime` exposes the available and
selected providers.

The portable build is CPU-only by default. `activate_gpu_runtime()` looks for an
optional `gpu_runtime/` directory next to the exe (or `OTR_GPU_RUNTIME_DIR`),
prepends its CUDA `onnxruntime` package to `sys.path`, and adds
`gpu_runtime/onnxruntime/capi` to the DLL search path before the first
`import onnxruntime`. Entry points call it at process startup; if the runtime
was already imported, it logs a warning and leaves the CPU path intact.
`scripts/setup_gpu_runtime.ps1` builds the add-on and
`scripts/verify_gpu.py` checks provider selection and CPU/GPU output parity.

## Translation contract

Request payload (abridged):

```json
{
  "scenario": "manga",
  "source_language": "Japanese",
  "target_language": "English",
  "reading_order": "rtl",
  "style_guidance": "Manga/webtoon dialogue...",
  "page_text": "旅人よ\n……",
  "items": [
    {
      "id": 0,
      "text": "旅人よ",
      "box": {"width": 42, "height": 190, "direction": "vertical"},
      "max_chars": 9,
      "context_after": "……"
    }
  ]
}
```

`page_text` is included whenever the page is split across several requests, and
chunk-boundary items also carry `context_before`/`context_after`. This keeps
names, pronouns, and a sentence that spans a batch boundary intact.

The system prompt enforces one output per id, concise localization, and a strict
JSON shape. Parsing:

1. Strip markdown fences if present.
2. Try `json.loads`; otherwise scan for the first balanced object/array while
   respecting strings and escapes.
3. Accept `{"items": [...]}`, a bare array, or an id → string mapping.
4. Accept `translation`, `translated_text`, `target`, or `text` fields.
5. Reconcile ids against the expected set.
6. Retry on total parse failure; fill missing ids with source text.

## Typesetting details

- **Font selection:** explicit `OTR_FONT_PATH`, project fonts, Microsoft YaHei,
  SimHei, Noto Sans CJK, PingFang, Arial Unicode.
- **Horizontal fit:** descending pixel sizes, wrap at each candidate, measure
  with `font.getlength`, require both width and height to fit.
- **Vertical fit:** one glyph per cell, columns filled top-to-bottom and drawn
  right-to-left; only used when the target is CJK and the source box is vertical.
- **Line breaking:** regex tokens for Latin words/numbers, CJK glyphs, spaces,
  and newlines. Closing punctuation is forbidden at line start; opening
  punctuation is moved off line end.
- **Colors:** border-majority background, high-contrast quartile foreground,
  WCAG contrast check and black/white repair below 3.2:1.
- **Overflow:** blocks expose `overflow` and `fit_ratio`. The UI flags them rather
  than silently clipping.

## Extension points

- Add an OCR engine by implementing
  `OCRBackend.read(image, **language_hints) -> OcrOutput`.
- Add an inpainter by implementing `Inpainter.inpaint(image, mask)`.
- Add a translation provider by implementing `Translator.translate()` and
  `aclose()`.
- Add rendering targets by consuming `RenderBlock` (the API already exposes box,
  font size, line height, lines, colors, and layout for an HTML Canvas overlay).
- Replace the in-process queue with Redis/Celery by matching `JobManager.submit`
  and `get` semantics.
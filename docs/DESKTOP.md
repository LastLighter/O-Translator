# Desktop client

The desktop app is a PySide6 assistant layered over the FastAPI pipeline. It
supports three capture modes, a tray presence, a global hotkey, and a floating
translation HUD.

```powershell
python -m client.main
# backwards-compatible:
python -m client.desktop
```

## Modes

| Mode | Trigger | Behavior |
|---|---|---|
| Manual upload | Mode tab, file picker, drag/drop, `Ctrl+V` | Sends a local image through the normal async job API |
| Area hotkey capture | Global hotkey, tray action, mode tab | DPI-aware region selection, then automatic job + floating result |
| Region auto-monitor | Mode tab / tray | Locks a region, detects meaningful changes, debounces, and auto-translates |

## Startup

The packaged exe and `python -m client.main` start in the system tray without
showing the main workbench. The backend starts in the background and a tray
balloon confirms that the app is ready; `F2` captures and translation results
appear in the floating HUD. Open the workbench from the tray menu
(**打开主窗口**) or by double-clicking the tray icon, or launch the exe again.
For development, `O-Translator.exe --show` (or `python launcher.py --show`)
opens the main window directly.

The hotkey defaults to `F2` and can be changed in the client with the
`QKeySequenceEdit` field. The setting persists in `QSettings`; installs that
still hold the previous `Ctrl + Alt + T` default are migrated to `F2` once. If
the global hotkey is occupied, the button/tray actions still work.

## Area capture

- Every connected screen gets its own overlay pane. A single window
  spanning mixed-DPI monitors can be scaled incorrectly; per-screen panes keep
  each screen's DPI mapping exact.
- The hotkey cursor position is captured before the main window hides. The first
  overlay frame already highlights the window/control under that point, even
  before the asynchronous UIA pass finishes.
- Move the pointer and the overlay highlights the innermost window/control under
  it. A plain left click or `Space` locks that highlighted rectangle; it does not
  start the job immediately.
- The wheel (or `Tab` / `Shift+Tab`) walks the detected hierarchy: control →
  client area → window → whole screen. The badge shows the current layer and its
  size.
- Drag with the left button to ignore automatic detection and draw a manual
  selection. Releasing the drag locks the rectangle and reveals the action bar.
- Once locked, the action bar offers `翻译` (confirm), `保存` (save the frozen
  crop as PNG), and `退出`. It is placed outside the selection on the side with
  the most free room; when the rectangle nearly touches every screen edge it
  moves inside. Left-dragging inside the rectangle moves the whole selection.
- A right click steps back one state: locked selection → free window/control
  selection, and free selection → close. `Esc` follows the same rule. `Enter`
  locks the whole active screen (press it again to confirm).
- Selections are clamped to the screen where the pointer/drag started.
- The selection's logical coordinates are paired with the matching mss physical
  monitor, then converted using measured `physical/logical` scale factors.
- A successful capture never re-opens the main workbench. The main window is
  hidden during selection and stays hidden afterwards; the translated image
  appears in the floating HUD instead. Open the main window from the tray only
  when you want the workbench.

### Coordinate pipeline

```
QScreen.geometry() (logical DIPs)
       │  pair by size + origin with mss.monitors (physical px)
       ▼
ScreenTarget(logical_rect, physical_rect, scale_x, scale_y)
       │
       ├─ overlay pane geometry = logical_rect
       ├─ drag selection = logical_rect-local
       ├─ Win32/UIA rects = physical px
       │      └─ physical_rect_to_logical() → logical rect in the pane
       └─ CaptureRegion = physical rect for mss + logical rect for HUD placement
```

`tests/test_screen_mapping.py` covers fractional DPI, negative virtual origins,
screen clamping, and two-monitor pairing. A live colored-quadrant test verified
1:1 crop alignment on the available screen.

## Automatic window/control picking

The snipper behaves like Snipaste: it highlights the window or control under the
pointer, locks it with a click or drag, then shows a confirmation toolbar. The
captured frame is frozen when the overlay opens, so painting and the final crop
use the same pixels.

### Modules

| Module | Responsibility |
|---|---|
| `client/services/ui_target.py` | `PhysicalRect`, `UITarget`, clipping/dedup/sort rules, `TargetStack` selection state |
| `client/services/win32_targets.py` | Fast z-order walk with `EnumWindows` / `GetWindowRect` / `GetClientRect` / child-window descent |
| `client/services/uia_targets.py` | Minimal raw-`ctypes` UI Automation COM client (no `comtypes` dependency) |
| `client/services/target_detector.py` | Daemon worker, one-slot request mailbox, throttling, UIA cache, Qt signals |
| `client/ui/overlay_snipper.py` | Highlight rendering, wheel/click/Space/drag interaction, physical↔logical mapping |

### Why not `ElementFromPoint`?

The overlay itself is an always-on-top window. `IUIAutomation::ElementFromPoint`
would return that overlay (or its Qt child), not the application below it. The
Win32 probe therefore walks the top-level z-order with `EnumWindows`, skips every
HWND owned by this process (plus cloaked, minimized, zero-alpha and click-through
helper windows), and picks the first visible window whose rectangle contains the
physical cursor position. The selected HWND becomes the seed for UI Automation.

### Win32 pass (fast path)

1. `EnumWindows` in z-order; stop at the first eligible HWND containing the point.
2. Take `DwmGetWindowAttribute(DWMWA_EXTENDED_FRAME_BOUNDS)` when available, so
   invisible resize borders are not included; fall back to `GetWindowRect`.
3. Add the client rectangle from `GetClientRect` + `ClientToScreen` when it
   differs from the frame.
4. Descend immediate child windows with `GetWindow(GW_CHILD/GW_HWNDNEXT)`,
   choosing the smallest visible child containing the point.
5. Labels come from the top-level title, the process executable name, and a
   small class-name map (`Button` → 按钮, `Chrome_RenderWidgetHostHWND` → 网页内容, …).

This pass is normally sub-millisecond and is emitted first, so the highlight
appears while UI Automation is still working.

### UI Automation pass (rich path)

1. Start UIA from the Win32 HWND: `ElementFromHandle` (not `ElementFromPoint`).
   For Chromium/Electron, the renderer child HWND is used as the seed because the
   browser frame often exposes only one generic pane.
2. Use the cached Control View walker (`get_ControlViewWalker`) and descend:
   for every current element, enumerate its children once, read
   `CurrentBoundingRectangle`, keep the smallest child that contains the point
   and is no larger than its parent, then repeat.
3. Read `CurrentName`, `CurrentLocalizedControlType`, `CurrentClassName` and
   `CurrentNativeWindowHandle` only for elements on the chosen path.
4. Add every ancestor in the path to the target list, then stop at the leaf, a
   15-level depth cap, 220 visited elements, or a 300 ms budget.

All calls use raw COM vtable slots through `ctypes`; `IUIAutomation`,
`IUIAutomationElement` and `IUIAutomationTreeWalker` pointers are released with
`Release()`, and the worker owns an MTA (`CoInitializeEx(COINIT_MULTITHREADED)`)
apartment. No new Python package is required.

### Ordering and hit-testing rules

All native rectangles are physical virtual-desktop pixels (the process is
per-monitor DPI aware after Qt starts). `normalize_targets()`:

1. clips each rectangle to the screen under the pointer;
2. drops side < 4 px (splitter grips, borders, scrollbar arrows);
3. collapses identical rectangles reported by Win32 and UIA, preferring a
   labelled UI Automation element;
4. sorts by area ascending (smallest = most specific control);
5. caps the stack at 14 layers and appends the screen as the outermost fallback.

The overlay highlights index 0 by default, and that default follows each richer
detection pass: the Win32 frame highlights the coarse window first, then the UIA
pass moves the default selection down to the deepest control under the pointer.
The wheel changes the index without starting a new detection. A UIA enrichment
keeps an explicitly wheel-selected outer layer by rectangle; the next fresh Win32
detection (for example after moving the pointer) resets to the innermost layer.

### MouseMove data flow

```
Qt mouseMove (UI thread)
  └─ overlay.request_detection(pane, global_logical_point)
       ├─ map to physical px with that screen's scale factors
       ├─ serial += 1; store as the pane's latest serial
       └─ detector.request(...)  → one-slot mailbox + wake token
                                     │
                       daemon thread │ (≤60 Hz)
                                     ▼
              win32_targets.collect(point)      ~0.2–2 ms
                  └─ emit("win32") ────────────► UI thread paints window/control instantly
              select_uia_seed(scan)
              uia_targets.collect(seed, point)  ~5–80 ms, cached 55/300 ms
                  └─ emit("uia") ──────────────► UI thread paints richer element chain
```

The Qt signal is queued; stale results are dropped by serial. The UI thread
never calls Win32/UIA while painting or handling mouse events. If a target app
hangs inside a UIA provider, the worker may block, but the overlay, drag
selection, and capture remain responsive; the next run uses a fresh worker stop
token. Elevated windows that deny UIA access simply fall back to the Win32
window/control rectangles.

### Failure handling

| Situation | Behavior |
|---|---|
| UIA COM initialization fails | Notice emitted; Win32 layers still work |
| `ElementFromHandle` denied or timed out | Win32 chain remains; UIA skipped for that sample |
| Window disappears between probes | Cached list is replaced on the next serial; stale results ignored |
| Minimized/cloaked/zero-alpha overlay | Skipped while walking the z-order |
| Non-Windows or headless Qt platform | Auto detection disabled; drag selection, toolbar confirmation and `Enter` still work |
| Detection rectangle outside the pane | Clipped to that screen; invalid rect reports `failed` instead of capturing |

### Performance notes

- The pointer position is coalesced into a single pending request, so a 125 Hz
  mouse cannot queue unbounded work. The overlay additionally caps requests at
  ~30 Hz and skips identical target chains.
- The desktop is captured **once** when the overlay opens. Each pane paints that
  frozen image, dims it, and restores the selected pixels from the same image,
  instead of compositing a full-screen translucent window on every mouse sample.
- Repaints are clipped to the union of the old and new selection rectangles (plus
  the size badge/toolbar), not the whole screen. The static instruction card is
  only redrawn when it intersects that dirty region.
- The Win32 pass is emitted before UIA, so the visible highlight is immediate;
  the first frame is primed synchronously with a single Win32 probe carried out at
  the hotkey cursor position.
- UIA results are reused for 55 ms, and for up to 300 ms while the pointer stays
  inside the same innermost element.
- COM calls run in an MTA daemon thread; `ctypes` releases the GIL, so the Qt
  event loop stays responsive.
- Confirming a selection crops the frozen frame directly, avoiding the old
  110 ms hide-and-grab compositor delay. The live mss path remains as a fallback
  when freezing the desktop failed.

## Region monitoring

`client/services/region_monitor.py` runs a `QThread` with its own `mss` handle.

1. Capture only the locked physical region.
2. Convert to grayscale and downscale to at most `diff_max_side` (480 px).
3. Ignore pixel changes below `noise_threshold` and changes smaller than
   `min_changed_pixels`.
4. Compare against the processed baseline to get a changed-pixel percentage.
5. If the percentage crosses `threshold_percent`, enter pending state.
6. Compare consecutive frames. While frame delta is above `static_percent`, keep
   refreshing the "last activity" timestamp (fade/transition settling).
7. When the frame is static for `debounce_ms` and the `cooldown_ms` since the
   last request has elapsed, emit the frame.
8. The main window saves the frame and queues the normal `/api/jobs` job.

Defaults:

| Setting | Default | Range |
|---|---|---|
| Sensitivity | 5% | 1–30% |
| Poll interval | 200 ms | 100–1000 ms |
| Debounce | 400 ms | 150–1500 ms |
| Cooldown | 2000 ms | 500–10000 ms |

The monitor is paused automatically while a translation job is in flight
(`suspend_capture`) so the result HUD cannot feed back into the next frame.

### Feedback and interference protection

- The floating HUD is excluded from the diff rectangle when it overlaps the
  monitored region.
- The main window is excluded as well while visible.
- Starting monitoring hides the main window to the tray by default so it cannot
  cover a game/manga region. This is configurable.
- A floating HUD for auto-monitoring is placed over the monitored region, exactly
  like a manual capture, so the translation stays aligned with the game UI.

## Floating HUD

`client/ui/floating_window.py` is frameless, always-on-top, and `Tool`-flagged.

- Snaps its image area exactly over the captured region by default (no manual
  "snap" button, no minimum-size clamping, no rounded frame or letterbox bars).
  A new capture or region re-aligns automatically. Transparent margins around the
  frame hold a soft blue glow, so the captured pixels are still covered
  pixel-for-pixel.
- There is no hover toolbar. Drag the translated image itself to move the HUD;
  dragging detaches it from the capture region. Right-click the image for
  **保存图片 / 关闭浮窗** (Ctrl+S / Esc also work while the HUD has focus).
- Click-through follows the result type: region monitoring is click-through by
  default, while a manual area capture stays interactive so it can be dragged,
  saved, or closed. On Windows a native `WM_NCHITTEST` override returns
  `HTTRANSPARENT` for the whole HUD while monitoring; disable the tray action to
  interact with it. The resize grip stays usable when detached.
- Drag the bottom-right grip to resize and detach the overlay into a normal
  window (enable the tray action first if click-through is active).
- Updates in place during monitoring.
- Progressively updated: the client polls every 300 ms, displays the backend's
  pass-1 `preview.png` as soon as it appears, then cross-fades (180 ms) to the
  LaMa-refined `result.png`. The HUD is never blank while LaMa is running.
- `保存` exports the displayed translated image.

## Tray menu

- 框选区域并翻译
- 开始 / 重选自动监控
- 暂停 / 继续自动监控
- 停止自动监控
- 浮窗鼠标穿透
- 打开主窗口
- 退出

Closing the main window hides it to the tray; monitoring and the global hotkey
continue. Use **退出** to terminate the backend process if the client started it.

### Single instance

Launching the same executable again does not create a second tray icon, overlay
stack, hotkey registration, floating HUD, or backend process. `SingleInstanceGuard`
uses a per-user `QLockFile` plus a `QLocalServer` named pipe scoped to the
application root. The second process sends an activation notification and exits;
the running instance restores its main window. Different builds (for example a
development checkout and a packaged release) use different scopes and may run
side by side.

## Persistence

`QSettings("O-Translator", "Desktop")` stores:

- language/scenario/reading-order options
- hotkey
- app mode
- monitor region (logical + physical + scale metadata)
- monitor sensitivity/poll/debounce/cooldown
- auto-overlay and auto-minimize toggles

On load, a saved monitor region is re-mapped against the current screen layout.
If it no longer intersects a screen, it is discarded.

## Dependencies

`PySide6`, `mss`, `opencv` (already in the core requirements), and `httpx`.
Automatic window/control picking uses only Windows system DLLs through
`ctypes` (`user32`, `dwmapi`, `shcore`, `ole32`, `oleaut32`); it does **not**
require `comtypes`, `uiautomation`, `pywin32`, or a UIAutomationClient .NET
assembly. The packaged build therefore needs no extra PyInstaller hidden import
or generated COM wrapper.

```powershell
pip install -r requirements-desktop.txt
```

## Performance notes

- Monitoring uses `mss`, grayscale conversion, and a downscaled diff. No OCR,
  network, PNG encoding, or full-resolution diffing happens until a trigger.
- Polling sleeps in the worker thread; pausing sleeps without capture.
- Translation work is dispatched through the existing bounded job queue.
- The client polls the job record at 300 ms while a job is active, so pass-1
  previews appear promptly. The preview and final image downloads are small PNGs
  fetched on a `QThreadPool` worker; the UI thread only swaps pixmaps.
- Optional GPU: extract the `gpu_runtime\` CUDA add-on next to
  `O-Translator.exe` and restart. The backend launches with the CUDA ONNX
  Runtime, selects `CUDAExecutionProvider`, and falls back to CPU if the driver
  cannot initialize. `OTR_DISABLE_GPU=1` forces CPU.
- Use a small region for the lowest CPU; very large high-refresh regions naturally
  cost more to capture.

## Windows packaging

`O-Translator.spec` builds a portable, double-clickable client:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows_exe.ps1
```

Close any running `dist\O-Translator\O-Translator.exe` before rebuilding: its
open `data\desktop-backend.log` blocks PyInstaller's replace step on Windows
(`WinError 32`). If the old instance must stay open, build to a separate tree
with `-DistPath dist-new` and use `dist-new\O-Translator`.

- One exe, two modes: `O-Translator.exe` runs the UI; `O-Translator.exe
  --backend` runs the local FastAPI service. `BackendManager` spawns the second
  form when the health probe fails, so packaged users never need a Python
  installation or a separate shell.
- `launcher.py` is the PyInstaller entry point. It repairs `sys.stdout`/`stderr`
  for `console=False` builds and forces UTF-8 so translated text cannot crash
  the log handlers.
- Frozen paths: writable `data/` and `models/` live next to the exe
  (`_app_root()`), while read-only resources (web UI, QSS, bundled LaMa and
  PP-OCRv4/v5 graphs) are read from `_internal` (`_bundle_root()` /
  `sys._MEIPASS`).
- `DEEPSEEK_API_KEY` is still the only key source. Because Explorer keeps a
  stale environment after `setx`, frozen backends also read the persisted value
  from `HKCU\Environment` when the process environment lacks it.
- RapidOCR and its ONNX Runtime backend are collected with
  `collect_data_files`/`collect_dynamic_libs`; the torch, TensorRT, OpenVINO,
  Paddle, and MNN inference engines are excluded from the build. PP-OCRv6 small
  ships inside the wheel, so default OCR works offline. PP-OCRv4/v5 graphs
  placed in `models/rapidocr/` are bundled under `_internal/models/rapidocr/`
  and copied into the writable model directory on first start.
- `scripts/smoke_test_exe.ps1` starts the bundled backend, submits a manga
  sample, and asserts OCR + DeepSeek translation completed.
- `O-Translator.exe --doctor` prints the frozen environment, dependency
  metadata, ONNX Runtime providers, and a RapidOCR initialization traceback for
  field diagnostics.
- `scripts/package_zip.py` writes `dist\O-Translator-windows-x64.zip` with
  forward-slash entry names; the PowerShell zip helpers can emit `\` names that
  some extractors mishandle. Use `--exclude data --exclude models` when
  re-packing a tree that has already been launched.
- `scripts/verify_dist.py <dir|zip>` checks that a release contains the exe,
  `_internal/`, LaMa, RapidOCR det+rec, and the `gpu_runtime/` CUDA add-on. A
  package is not ready to ship until this command reports `PASS`.

`OTR_BUNDLE_OCR=0` produces a smaller UI-only build with OCR disabled; it is
useful for checking Qt plugins and spawn behaviour quickly. The old
`OTR_BUNDLE_PADDLE` / `-NoPaddle` names are still accepted as aliases.
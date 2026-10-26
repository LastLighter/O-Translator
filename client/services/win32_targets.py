"""Fast Win32 window/control probing for the snipping overlay.

UI Automation's ``ElementFromPoint`` cannot be used while our own topmost
overlay is on screen: it correctly reports the overlay, not the application the
user wants to capture.  This module therefore walks the Win32 z-order itself,
skips windows owned by this process, and descends the child-window tree below
the cursor.  It is deliberately small and fast (typically well under a
millisecond per mouse sample) and doubles as the HWND seed for the slower UI
Automation pass.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from pathlib import Path

from client.services.ui_target import (
    CLIENT,
    CONTROL,
    WINDOW,
    PhysicalRect,
    UITarget,
    WindowScan,
)

IS_WINDOWS = os.name == "nt"

if IS_WINDOWS:  # pragma: no branch - imported on non-Windows test hosts too
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    try:
        _dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
    except OSError:  # pragma: no cover - stripped Windows
        _dwmapi = None
    try:
        _shcore = ctypes.WinDLL("shcore", use_last_error=True)
    except OSError:  # pragma: no cover
        _shcore = None
else:  # pragma: no cover - exercised only on non-Windows hosts
    _user32 = None
    _kernel32 = None
    _dwmapi = None
    _shcore = None

HRESULT = ctypes.c_long
_FUNCTYPE = ctypes.WINFUNCTYPE if IS_WINDOWS else ctypes.CFUNCTYPE
WNDENUMPROC = _FUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

_GW_CHILD = 5
_GW_HWNDNEXT = 2
_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_LAYERED = 0x00080000
_LWA_ALPHA = 0x00000002
_DWMWA_EXTENDED_FRAME_BOUNDS = 9
_DWMWA_CLOAKED = 14
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

_KIND_CLASS_LABELS = {
    "Chrome_RenderWidgetHostHWND": "网页内容",
    "Chrome_WidgetWin_1": "浏览器窗口",
    "MozillaWindowClass": "浏览器窗口",
    "ApplicationFrameWindow": "应用窗口",
    "Edit": "输入框",
    "RichEdit20W": "文本编辑",
    "RichEditD2DPT": "文本编辑",
    "RICHEDIT50W": "文本编辑",
    "Button": "按钮",
    "Static": "文本",
    "ComboBox": "下拉框",
    "ComboBoxEx32": "下拉框",
    "SysListView32": "列表",
    "SysTreeView32": "树",
    "SysTabControl32": "标签页",
    "DirectUIHWND": "界面",
    "Intermediate D3D Window": "画面",
    "Shell_TrayWnd": "任务栏",
    "Progman": "桌面",
    "WorkerW": "桌面",
    "#32770": "对话框",
}

# Chromium/Electron expose their DOM tree below the renderer HWND, not below
# the browser frame.  Seeding UIA from the renderer produces useful names for
# web pages; seeding from the frame often yields only one generic pane.
_WEB_RENDERER_CLASSES = {
    "Chrome_RenderWidgetHostHWND",
    "Chrome_RenderWidgetHostHWND1",
    "Internet Explorer_Server",
}
_WEB_HOST_CLASSES = {
    "Chrome_WidgetWin_0",
    "Chrome_WidgetWin_1",
    "CefBrowserWindow",
    "WebViewHost",
}

_process_names: dict[int, str] = {}


def _configure() -> None:
    """Declare ctypes signatures once (no-op off Windows)."""
    if not IS_WINDOWS or _user32 is None:  # pragma: no cover
        return
    user32 = _user32
    user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetWindow.restype = wintypes.HWND
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.ClientToScreen.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClassNameW.restype = ctypes.c_int
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetLayeredWindowAttributes.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.COLORREF),
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.GetLayeredWindowAttributes.restype = wintypes.BOOL

    long_ptr = ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long
    get_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
    get_long.argtypes = [wintypes.HWND, ctypes.c_int]
    get_long.restype = long_ptr
    globals()["_get_window_long"] = get_long

    if _dwmapi is not None:
        _dwmapi.DwmGetWindowAttribute.argtypes = [
            wintypes.HWND,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        _dwmapi.DwmGetWindowAttribute.restype = HRESULT

    if _shcore is not None:
        _shcore.GetProcessDpiAwareness.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        _shcore.GetProcessDpiAwareness.restype = HRESULT

    kernel32 = _kernel32
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL


def _call_configured() -> None:
    if _user32 is not None and "_get_window_long" not in globals():
        _configure()


_call_configured()


def process_dpi_awareness() -> int:
    """Return 0/1/2 for unaware/system/per-monitor DPI awareness."""
    if not IS_WINDOWS or _shcore is None:  # pragma: no cover
        return 0
    value = ctypes.c_int(0)
    try:
        if _shcore.GetProcessDpiAwareness(None, ctypes.byref(value)) >= 0:
            return int(value.value)
    except OSError:  # pragma: no cover
        pass
    return 0


def _ex_style(hwnd: int) -> int:
    try:
        return int(_get_window_long(hwnd, _GWL_EXSTYLE) or 0) & 0xFFFFFFFF
    except (OSError, NameError):  # pragma: no cover
        return 0


def _is_cloaked(hwnd: int) -> bool:
    if _dwmapi is None:  # pragma: no cover
        return False
    cloaked = ctypes.c_int(0)
    try:
        hr = _dwmapi.DwmGetWindowAttribute(
            hwnd,
            _DWMWA_CLOAKED,
            ctypes.byref(cloaked),
            ctypes.sizeof(cloaked),
        )
        return hr >= 0 and cloaked.value != 0
    except OSError:  # pragma: no cover
        return False


def _is_fully_transparent(hwnd: int) -> bool:
    if not (_ex_style(hwnd) & _WS_EX_LAYERED) or _user32 is None:
        return False
    key = wintypes.COLORREF(0)
    alpha = ctypes.c_ubyte(255)
    flags = wintypes.DWORD(0)
    try:
        if _user32.GetLayeredWindowAttributes(
            hwnd, ctypes.byref(key), ctypes.byref(alpha), ctypes.byref(flags)
        ):
            return bool(flags.value & _LWA_ALPHA) and alpha.value == 0
    except OSError:  # pragma: no cover
        pass
    return False


def _window_rect(hwnd: int, *, extended_frame: bool = False) -> PhysicalRect:
    rect = wintypes.RECT()
    if extended_frame and _dwmapi is not None:
        try:
            hr = _dwmapi.DwmGetWindowAttribute(
                hwnd,
                _DWMWA_EXTENDED_FRAME_BOUNDS,
                ctypes.byref(rect),
                ctypes.sizeof(rect),
            )
            if hr >= 0 and rect.right > rect.left and rect.bottom > rect.top:
                return PhysicalRect.from_ltrb(
                    rect.left, rect.top, rect.right, rect.bottom
                )
        except OSError:  # pragma: no cover
            pass
    if not _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return PhysicalRect(0, 0, 0, 0)
    return PhysicalRect.from_ltrb(rect.left, rect.top, rect.right, rect.bottom)


def _client_rect(hwnd: int) -> PhysicalRect:
    rect = wintypes.RECT()
    if not _user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return PhysicalRect(0, 0, 0, 0)
    width = max(0, rect.right - rect.left)
    height = max(0, rect.bottom - rect.top)
    if width <= 0 or height <= 0:
        return PhysicalRect(0, 0, 0, 0)
    origin = wintypes.POINT(0, 0)
    if not _user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return PhysicalRect(0, 0, 0, 0)
    return PhysicalRect(origin.x, origin.y, width, height)


def _class_name(hwnd: int) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    try:
        length = _user32.GetClassNameW(hwnd, buffer, len(buffer))
    except OSError:  # pragma: no cover
        return ""
    if length <= 0:
        return ""
    return buffer.value


def _window_text(hwnd: int) -> str:
    try:
        length = _user32.GetWindowTextLengthW(hwnd)
        if length <= 0 or length > 4096:
            return ""
        buffer = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buffer, len(buffer))
        return buffer.value.strip()
    except OSError:  # pragma: no cover
        return ""


def _window_pid(hwnd: int) -> int:
    pid = wintypes.DWORD(0)
    try:
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    except OSError:  # pragma: no cover
        return 0
    return int(pid.value)


def process_name(pid: int) -> str:
    """Executable stem for a process id, cached and failure-tolerant."""
    if not pid or not IS_WINDOWS or _kernel32 is None:  # pragma: no cover
        return ""
    cached = _process_names.get(pid)
    if cached is not None:
        return cached
    name = ""
    handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if handle:
        try:
            buffer = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(len(buffer))
            if _kernel32.QueryFullProcessImageNameW(
                handle, 0, buffer, ctypes.byref(size)
            ):
                name = Path(buffer.value).stem
        except OSError:  # pragma: no cover
            name = ""
        finally:
            _kernel32.CloseHandle(handle)
    if len(_process_names) > 256:
        _process_names.clear()
    _process_names[pid] = name
    return name


def _label_for_window(hwnd: int, class_name: str, app: str) -> str:
    title = _window_text(hwnd)
    if title:
        if app and app.casefold() not in title.casefold():
            return f"{title} · {app}"
        return title
    return _KIND_CLASS_LABELS.get(class_name, class_name or app or "窗口")


def _class_label(class_name: str) -> str:
    return _KIND_CLASS_LABELS.get(class_name, class_name or "控件")


def _is_top_level_candidate(hwnd: int, x: int, y: int, own_pid: int) -> bool:
    if not _user32.IsWindowVisible(hwnd) or _user32.IsIconic(hwnd):
        return False
    if _window_pid(hwnd) == own_pid:
        return False
    ex_style = _ex_style(hwnd)
    if ex_style & _WS_EX_TRANSPARENT:
        # Click-through helper/HUD windows are not the visual surface below
        # the pointer; keep walking down the z-order.
        return False
    if _is_cloaked(hwnd) or _is_fully_transparent(hwnd):
        return False
    rect = _window_rect(hwnd)
    return rect.width > 1 and rect.height > 1 and rect.contains(x, y)


def _topmost_window_at(x: int, y: int, own_pid: int) -> int:
    if not IS_WINDOWS or _user32 is None:  # pragma: no cover
        return 0
    found: list[int] = []

    @WNDENUMPROC
    def _callback(hwnd: int, _lparam: int) -> bool:
        if _is_top_level_candidate(int(hwnd), x, y, own_pid):
            found.append(int(hwnd))
            return False
        return True

    try:
        _user32.EnumWindows(_callback, 0)
    except OSError:  # pragma: no cover
        return 0
    return found[0] if found else 0


def _child_at(hwnd: int, x: int, y: int, own_pid: int) -> int:
    """Smallest immediate child containing the point, or 0."""
    best = 0
    best_area = 1 << 62
    try:
        child = _user32.GetWindow(hwnd, _GW_CHILD)
    except OSError:  # pragma: no cover
        return 0
    guard = 0
    while child and guard < 512:
        guard += 1
        try:
            next_child = _user32.GetWindow(child, _GW_HWNDNEXT)
        except OSError:  # pragma: no cover
            return best
        if (
            _user32.IsWindowVisible(child)
            and not _user32.IsIconic(child)
            and _window_pid(child) != own_pid
        ):
            rect = _window_rect(child)
            if rect.contains(x, y) and rect.area < best_area:
                best = int(child)
                best_area = rect.area
        child = next_child
    return best


def collect(point: tuple[int, int], *, own_pid: int | None = None) -> WindowScan:
    """Return the Win32 target chain below a physical screen point."""
    if not IS_WINDOWS or _user32 is None:  # pragma: no cover
        return WindowScan()
    x, y = int(point[0]), int(point[1])
    own_pid = os.getpid() if own_pid is None else int(own_pid)
    hwnd = _topmost_window_at(x, y, own_pid)
    if not hwnd:
        return WindowScan()

    pid = _window_pid(hwnd)
    app = process_name(pid)
    class_name = _class_name(hwnd)
    targets: list[UITarget] = []
    frame = _window_rect(hwnd, extended_frame=True)
    if frame.area > 0:
        targets.append(
            UITarget(
                rect=frame,
                label=_label_for_window(hwnd, class_name, app),
                kind=WINDOW,
                source="win32",
                hwnd=hwnd,
                depth=0,
            )
        )
    client = _client_rect(hwnd)
    if client.area > 0 and (
        abs(client.left - frame.left) > 2
        or abs(client.top - frame.top) > 2
        or abs(client.width - frame.width) > 2
        or abs(client.height - frame.height) > 2
    ):
        targets.append(
            UITarget(
                rect=client,
                label="客户区",
                kind=CLIENT,
                source="win32",
                hwnd=hwnd,
                depth=1,
            )
        )

    current = hwnd
    deep_class = ""
    for depth in range(2, 12):
        child = _child_at(current, x, y, own_pid)
        if not child:
            break
        rect = _window_rect(child)
        if rect.area <= 0:
            break
        child_class = _class_name(child)
        targets.append(
            UITarget(
                rect=rect,
                label=_class_label(child_class),
                kind=CONTROL,
                source="win32",
                hwnd=child,
                depth=depth,
            )
        )
        current = child
        deep_class = child_class

    return WindowScan(
        targets=tuple(targets),
        top_hwnd=hwnd,
        pid=pid,
        app=app,
        top_class=class_name,
        deep_hwnd=current if current != hwnd else 0,
        deep_class=deep_class,
    )


def is_desktop_window(class_name: str) -> bool:
    return class_name in {"Progman", "WorkerW"}


def select_uia_seed(scan: WindowScan) -> int:
    """Pick the HWND UI Automation should start from.

    Web renderers expose the page tree below their child HWND.  Native apps are
    best served by their top-level window so UIA can see the full control tree.
    """
    if scan.deep_hwnd and (
        scan.deep_class in _WEB_RENDERER_CLASSES
        or scan.top_class in _WEB_HOST_CLASSES
    ):
        return scan.deep_hwnd
    return scan.top_hwnd
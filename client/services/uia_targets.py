"""Minimal UI Automation client for the snipping overlay.

Uses raw ``ctypes`` COM bindings instead of ``comtypes``/``uiautomation`` so the
desktop build gains no new runtime dependency and PyInstaller has no generated
wrapper modules to discover.  The client is created and used on the detector's
background thread only.

Why not ``ElementFromPoint``?  The selection overlay is always-on-top, so UIA
would faithfully report our own pane.  We instead seed UIA from the HWND found
by :mod:`client.services.win32_targets` and descend the Control View tree,
choosing the smallest child rectangle that contains the cursor.  This also
yields the nested ancestor rectangles for free.
"""

from __future__ import annotations

import ctypes
import os
import time
import uuid
from contextlib import suppress
from ctypes import wintypes
from typing import Any

from client.services.ui_target import (
    ELEMENT,
    WINDOW,
    PhysicalRect,
    UITarget,
)

IS_WINDOWS = os.name == "nt"

if IS_WINDOWS:  # pragma: no branch - imported on non-Windows hosts too
    _ole32 = ctypes.WinDLL("ole32", use_last_error=True)
    _oleaut32 = ctypes.WinDLL("oleaut32", use_last_error=True)
else:  # pragma: no cover - exercised only on non-Windows hosts
    _ole32 = None
    _oleaut32 = None

HRESULT = ctypes.c_long
_FUNCTYPE = ctypes.WINFUNCTYPE if IS_WINDOWS else ctypes.CFUNCTYPE

# IUIAutomation vtable slots (IUnknown occupies 0..2).
_SLOT_RELEASE = 2
_SLOT_ELEMENT_FROM_HANDLE = 6
_SLOT_GET_CONTROL_VIEW_WALKER = 14

# IUIAutomationTreeWalker vtable slots.
_SLOT_WALKER_FIRST_CHILD = 4
_SLOT_WALKER_NEXT_SIBLING = 6

# IUIAutomationElement vtable slots.
_SLOT_CURRENT_CONTROL_TYPE = 21
_SLOT_CURRENT_LOCALIZED_CONTROL_TYPE = 22
_SLOT_CURRENT_NAME = 23
_SLOT_CURRENT_CLASS_NAME = 30
_SLOT_CURRENT_NATIVE_HWND = 36
_SLOT_CURRENT_BOUNDING_RECT = 43

_CLSID_CUIAUTOMATION = "ff48dba4-60ef-4201-aa87-54103eef594e"
_IID_IUIAUTOMATION = "30cbe57d-d9d0-452a-ab13-7ac5ac4825ee"
_CLSCTX_INPROC_SERVER = 1
_COINIT_MULTITHREADED = 0
_RPC_E_CHANGED_MODE = 0x80010106

_RELEASE_PROTO = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
_ELEMENT_FROM_HANDLE_PROTO = ctypes.WINFUNCTYPE(
    HRESULT, ctypes.c_void_p, wintypes.HWND, ctypes.POINTER(ctypes.c_void_p)
)
_GET_WALKER_PROTO = ctypes.WINFUNCTYPE(
    HRESULT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
)
_GET_STRING_PROTO = ctypes.WINFUNCTYPE(
    HRESULT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
)
_GET_INT_PROTO = ctypes.WINFUNCTYPE(
    HRESULT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)
)
_GET_HWND_PROTO = ctypes.WINFUNCTYPE(
    HRESULT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
)
_GET_RECT_PROTO = ctypes.WINFUNCTYPE(
    HRESULT, ctypes.c_void_p, ctypes.POINTER(wintypes.RECT)
)
_WALKER_CHILD_PROTO = ctypes.WINFUNCTYPE(
    HRESULT,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
)


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    def __init__(self, text: str) -> None:
        value = uuid.UUID(text)
        super().__init__(
            value.time_low,
            value.time_mid,
            value.time_hi_version,
            (ctypes.c_ubyte * 8)(*value.bytes[8:]),
        )


def _configure() -> None:
    if not IS_WINDOWS or _ole32 is None:  # pragma: no cover
        return
    _ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    _ole32.CoInitializeEx.restype = HRESULT
    _ole32.CoUninitialize.argtypes = []
    _ole32.CoUninitialize.restype = None
    _ole32.CoCreateInstance.argtypes = [
        ctypes.POINTER(GUID),
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(GUID),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _ole32.CoCreateInstance.restype = HRESULT
    _oleaut32.SysFreeString.argtypes = [ctypes.c_void_p]
    _oleaut32.SysFreeString.restype = None


_configure()


def _vtable(pointer: int | ctypes.c_void_p) -> Any:
    address = pointer.value if isinstance(pointer, ctypes.c_void_p) else int(pointer)
    return ctypes.cast(
        address, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
    ).contents


def _invoke(
    pointer: int | ctypes.c_void_p,
    slot: int,
    prototype: Any,
    *args: Any,
) -> Any:
    function = prototype(_vtable(pointer)[slot])
    return function(pointer, *args)


def _release(pointer: int | ctypes.c_void_p | None) -> None:
    if not pointer:
        return
    with suppress(OSError):  # pragma: no cover - broken provider
        _invoke(pointer, _SLOT_RELEASE, _RELEASE_PROTO)


def _get_rect(element: int) -> PhysicalRect | None:
    rect = wintypes.RECT()
    try:
        hr = _invoke(
            element, _SLOT_CURRENT_BOUNDING_RECT, _GET_RECT_PROTO, ctypes.byref(rect)
        )
    except OSError:
        return None
    if hr < 0:
        return None
    if rect.right <= rect.left or rect.bottom <= rect.top:
        return None
    return PhysicalRect.from_ltrb(rect.left, rect.top, rect.right, rect.bottom)


def _get_int(element: int, slot: int) -> int:
    value = ctypes.c_int(0)
    try:
        hr = _invoke(element, slot, _GET_INT_PROTO, ctypes.byref(value))
    except OSError:
        return 0
    return int(value.value) if hr >= 0 else 0


def _get_hwnd(element: int) -> int:
    value = ctypes.c_void_p()
    try:
        hr = _invoke(
            element, _SLOT_CURRENT_NATIVE_HWND, _GET_HWND_PROTO, ctypes.byref(value)
        )
    except OSError:
        return 0
    return int(value.value or 0) if hr >= 0 else 0


def _get_string(element: int, slot: int) -> str:
    value = ctypes.c_void_p()
    try:
        hr = _invoke(element, slot, _GET_STRING_PROTO, ctypes.byref(value))
    except OSError:
        return ""
    if hr < 0 or not value.value:
        return ""
    try:
        return ctypes.wstring_at(value.value).strip()
    finally:
        if _oleaut32 is not None:
            _oleaut32.SysFreeString(value)


def _first_child(walker: int, element: int) -> int:
    value = ctypes.c_void_p()
    try:
        hr = _invoke(
            walker,
            _SLOT_WALKER_FIRST_CHILD,
            _WALKER_CHILD_PROTO,
            element,
            ctypes.byref(value),
        )
    except OSError:
        return 0
    return int(value.value or 0) if hr >= 0 else 0


def _next_sibling(walker: int, element: int) -> int:
    value = ctypes.c_void_p()
    try:
        hr = _invoke(
            walker,
            _SLOT_WALKER_NEXT_SIBLING,
            _WALKER_CHILD_PROTO,
            element,
            ctypes.byref(value),
        )
    except OSError:
        return 0
    return int(value.value or 0) if hr >= 0 else 0


class UIAClient:
    """Thread-affine Control View UIA client.

    Create one instance inside the detector thread, call :meth:`start` once,
    then :meth:`collect` for every sample.  Failures degrade to the Win32
    chain instead of raising into the UI.
    """

    def __init__(self) -> None:
        self._automation = 0
        self._walker = 0
        self._com_owned = False
        self._broken = False
        self.error = ""

    @property
    def available(self) -> bool:
        return bool(self._automation and self._walker and not self._broken)

    def start(self) -> bool:
        if not IS_WINDOWS or _ole32 is None:  # pragma: no cover
            self.error = "UI Automation is only available on Windows."
            return False
        if self._automation:
            return True
        try:
            hr = int(_ole32.CoInitializeEx(None, _COINIT_MULTITHREADED))
            if hr < 0 and (hr & 0xFFFFFFFF) != _RPC_E_CHANGED_MODE:
                self.error = f"CoInitializeEx failed: 0x{hr & 0xFFFFFFFF:08X}"
                return False
            self._com_owned = hr >= 0
            clsid = GUID(_CLSID_CUIAUTOMATION)
            iid = GUID(_IID_IUIAUTOMATION)
            automation = ctypes.c_void_p()
            hr = int(
                _ole32.CoCreateInstance(
                    ctypes.byref(clsid),
                    None,
                    _CLSCTX_INPROC_SERVER,
                    ctypes.byref(iid),
                    ctypes.byref(automation),
                )
            )
            if hr < 0 or not automation.value:
                self.error = f"CoCreateInstance(IUIAutomation) failed: 0x{hr & 0xFFFFFFFF:08X}"
                self._uninitialize()
                return False
            walker = ctypes.c_void_p()
            hr = int(
                _invoke(
                    automation.value,
                    _SLOT_GET_CONTROL_VIEW_WALKER,
                    _GET_WALKER_PROTO,
                    ctypes.byref(walker),
                )
            )
            if hr < 0 or not walker.value:
                _release(automation.value)
                self.error = f"ControlViewWalker unavailable: 0x{hr & 0xFFFFFFFF:08X}"
                self._uninitialize()
                return False
            self._automation = int(automation.value)
            self._walker = int(walker.value)
            return True
        except OSError as exc:
            self.error = f"UI Automation init crashed: {exc}"
            self._uninitialize()
            return False

    def stop(self) -> None:
        _release(self._walker)
        _release(self._automation)
        self._walker = 0
        self._automation = 0
        self._uninitialize()

    def _uninitialize(self) -> None:
        if self._com_owned and _ole32 is not None:
            with suppress(OSError):  # pragma: no cover
                _ole32.CoUninitialize()
        self._com_owned = False

    def _smallest_child(
        self,
        parent: int,
        parent_rect: PhysicalRect,
        x: int,
        y: int,
        *,
        visited: list[int],
        deadline: float,
    ) -> int:
        best = 0
        best_area = 1 << 62
        child = _first_child(self._walker, parent)
        while child:
            if (
                time.perf_counter() > deadline
                or visited[0] > 220
                or best_area <= 4
            ):
                _release(child)
                break
            visited[0] += 1
            next_child = _next_sibling(self._walker, child)
            rect = _get_rect(child)
            if (
                rect is not None
                and rect.width >= 2
                and rect.height >= 2
                and rect.area <= parent_rect.area
                and rect.contains(x, y)
                and rect.area < best_area
            ):
                _release(best)
                best = child
                best_area = rect.area
            else:
                _release(child)
            child = next_child
        return best

    def _describe(
        self,
        element: int,
        rect: PhysicalRect,
        *,
        root: bool,
        app: str,
        depth: int,
    ) -> UITarget:
        name = _get_string(element, _SLOT_CURRENT_NAME)
        localized = _get_string(
            element, _SLOT_CURRENT_LOCALIZED_CONTROL_TYPE
        )
        class_name = _get_string(element, _SLOT_CURRENT_CLASS_NAME)
        label = name or localized or class_name or (app if root else "元素")
        if root and app and app.casefold() not in label.casefold():
            label = f"{label} · {app}"
        return UITarget(
            rect=rect,
            label=label,
            kind=WINDOW if root else ELEMENT,
            source="uia",
            hwnd=_get_hwnd(element) if root else 0,
            depth=depth,
        )

    def collect(
        self,
        hwnd: int,
        point: tuple[int, int],
        *,
        app: str = "",
        top_class: str = "",
        time_budget: float = 0.30,
    ) -> list[UITarget]:
        if not self.available or not hwnd:
            return []
        # The desktop's UIA tree is enormous and never a useful capture target.
        if top_class in {"Progman", "WorkerW"}:
            return []
        x, y = int(point[0]), int(point[1])
        deadline = time.perf_counter() + max(0.05, float(time_budget))
        targets: list[UITarget] = []
        root = 0
        current = 0
        try:
            root_ptr = ctypes.c_void_p()
            hr = int(
                _invoke(
                    self._automation,
                    _SLOT_ELEMENT_FROM_HANDLE,
                    _ELEMENT_FROM_HANDLE_PROTO,
                    wintypes.HWND(hwnd),
                    ctypes.byref(root_ptr),
                )
            )
            if hr < 0 or not root_ptr.value:
                return []
            root = int(root_ptr.value)
            root_rect = _get_rect(root)
            if root_rect is None or not root_rect.contains(x, y):
                return []
            targets.append(
                self._describe(root, root_rect, root=True, app=app, depth=0)
            )
            current = root
            current_rect = root_rect
            visited = [0]
            for depth in range(1, 15):
                if time.perf_counter() > deadline or visited[0] > 220:
                    break
                child = self._smallest_child(
                    current,
                    current_rect,
                    x,
                    y,
                    visited=visited,
                    deadline=deadline,
                )
                if not child:
                    break
                child_rect = _get_rect(child)
                if child_rect is None:
                    _release(child)
                    break
                targets.append(
                    self._describe(
                        child,
                        child_rect,
                        root=False,
                        app=app,
                        depth=depth,
                    )
                )
                if current != root:
                    _release(current)
                current = child
                current_rect = child_rect
            return targets
        except OSError as exc:  # COM/provider crash: fall back to Win32 next time
            self._broken = True
            self.error = f"UI Automation query failed: {exc}"
            return targets
        finally:
            if current and current != root:
                _release(current)
            _release(root)

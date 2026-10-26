"""Background target detection for the snipping overlay.

Mouse movement must stay smooth even though UI Automation can take tens of
milliseconds on Chromium/Electron/Office windows.  The pipeline is therefore:

1. the UI thread only stores the newest request (a one-slot mailbox);
2. a daemon thread computes the fast Win32 chain, emits it, then enriches it
   with UI Automation;
3. stale results are discarded by serial number on the UI thread.

A short cache reuses UIA rectangles while the pointer stays inside the same
innermost element, so moving across a large control does not repeat expensive
cross-process calls.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from contextlib import suppress
from dataclasses import dataclass

from PySide6 import QtCore, QtGui

from client.services import uia_targets, win32_targets
from client.services.ui_target import (
    PhysicalRect,
    TargetStack,
    UITarget,
    WindowScan,
    normalize_targets,
)

_TOKEN = object()


@dataclass(frozen=True, slots=True)
class DetectRequest:
    serial: int
    pane_index: int
    x: int
    y: int
    screen_left: int
    screen_top: int
    screen_width: int
    screen_height: int
    screen_label: str


def _signature(targets: list[UITarget]) -> tuple[tuple[int, int, int, int, str, str], ...]:
    return tuple(
        (
            target.rect.left,
            target.rect.top,
            target.rect.width,
            target.rect.height,
            target.label,
            target.kind,
        )
        for target in targets
    )


def _containing(targets: list[UITarget], x: int, y: int) -> list[UITarget]:
    """Keep only rectangles that still contain the sampled pointer position.

    UIA results are cached for a few frames; without this filter a fast pointer
    move could briefly paint a stale child rectangle that no longer lies under
    the cursor.
    """
    return [target for target in targets if target.rect.contains(x, y)]


def native_targets_supported() -> bool:
    """Whether the fast Win32 z-order probe can run in this session."""
    return TargetDetector._detect_native_support()


def quick_targets(
    point: tuple[int, int],
    *,
    screen: PhysicalRect,
    screen_label: str,
) -> list[UITarget]:
    """One synchronous Win32 sample used to prime a freshly opened overlay.

    The regular detector runs asynchronously, which is right for mouse tracking
    but leaves a short gap after the hotkey is pressed where no highlight exists.
    This is a single sub-millisecond probe (no UIA), so the overlay can paint the
    window/control under the hotkey cursor on its very first frame.
    """
    if not native_targets_supported():
        return normalize_targets([], screen=screen, screen_label=screen_label)
    try:
        scan = win32_targets.collect((int(point[0]), int(point[1])), own_pid=os.getpid())
    except OSError:  # pragma: no cover - defensive
        scan = WindowScan()
    return normalize_targets(
        list(scan.targets), screen=screen, screen_label=screen_label
    )


class TargetDetector(QtCore.QObject):
    """Computes layered window/control rectangles on a background thread."""

    # serial, pane index, list[UITarget], stage ("win32"/"uia"/"screen")
    detected = QtCore.Signal(int, int, object, str)
    notice = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._queue: queue.Queue[object] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._pending: DetectRequest | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_stop: threading.Event | None = None
        self._native_enabled = self._detect_native_support()
        self._uia_hint = ""

    @staticmethod
    def _detect_native_support() -> bool:
        if not win32_targets.IS_WINDOWS:
            return False
        app = QtGui.QGuiApplication.instance()
        if app is not None:
            platform = QtGui.QGuiApplication.platformName().lower()
            if platform in {"offscreen", "minimal"}:
                # Unit tests and headless sessions have no real z-order to walk.
                return False
        return True

    @property
    def native_enabled(self) -> bool:
        return self._native_enabled

    @property
    def uia_hint(self) -> str:
        return self._uia_hint

    def start(self) -> None:
        if (
            self._thread is not None
            and self._thread.is_alive()
            and self._thread_stop is not None
            and not self._thread_stop.is_set()
        ):
            return
        # A new run gets its own stop event.  If a previous UIA call is still
        # blocking inside the old daemon thread, it will observe its own event
        # being set and exit without touching this run's results.
        stop_event = threading.Event()
        self._stop = stop_event
        self._thread_stop = stop_event
        self._thread = threading.Thread(
            target=self._run,
            args=(stop_event,),
            name="otr-target-detector",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        stop_event = self._thread_stop or self._stop
        stop_event.set()
        self._wake()
        thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=0.35)
        with self._lock:
            self._pending = None

    def _wake(self) -> None:
        # A token is already queued when this is full; the worker always reads
        # the newest request from the mailbox.
        with suppress(queue.Full):
            self._queue.put_nowait(_TOKEN)

    def request(
        self,
        *,
        serial: int,
        pane_index: int,
        x: int,
        y: int,
        screen_rect: PhysicalRect,
        screen_label: str,
    ) -> None:
        if self._stop.is_set():
            return
        request = DetectRequest(
            serial=int(serial),
            pane_index=int(pane_index),
            x=int(x),
            y=int(y),
            screen_left=int(screen_rect.left),
            screen_top=int(screen_rect.top),
            screen_width=int(screen_rect.width),
            screen_height=int(screen_rect.height),
            screen_label=str(screen_label),
        )
        with self._lock:
            self._pending = request
        self._wake()

    def _take_pending(self) -> DetectRequest | None:
        with self._lock:
            request = self._pending
            self._pending = None
        return request

    def _emit(
        self,
        request: DetectRequest,
        targets: list[UITarget],
        stage: str,
        stop_event: threading.Event,
    ) -> None:
        if stop_event.is_set():
            return
        try:
            self.detected.emit(request.serial, request.pane_index, targets, stage)
        except RuntimeError:  # pragma: no cover - receiver already destroyed
            stop_event.set()

    def _run(self, stop_event: threading.Event) -> None:
        uia = uia_targets.UIAClient()
        if self._native_enabled:
            if uia.start():
                self._uia_hint = "UI Automation 已启用"
            else:
                self._uia_hint = uia.error or "UI Automation 不可用"
                if not stop_event.is_set():
                    with suppress(RuntimeError):
                        self.notice.emit(
                            "UI Automation 不可用，已降级为 Win32 窗口级识别。"
                        )
        elif not stop_event.is_set():
            with suppress(RuntimeError):
                self.notice.emit("当前平台不支持自动识别，可拖拽框选。")

        last_cycle = 0.0
        last_uia: tuple[int, list[UITarget], float] | None = None
        last_uia_at = 0.0
        try:
            while not stop_event.is_set():
                try:
                    self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if stop_event.is_set():
                    break
                request = self._take_pending()
                if request is None:
                    continue

                # ~60 Hz ceiling: mouse move events can arrive much faster,
                # and the mailbox already keeps only the newest position.
                interval = time.perf_counter() - last_cycle
                if interval < 0.016 and stop_event.wait(0.016 - interval):
                    break
                last_cycle = time.perf_counter()

                screen = PhysicalRect(
                    request.screen_left,
                    request.screen_top,
                    request.screen_width,
                    request.screen_height,
                )
                scan = WindowScan()
                native: list[UITarget] = []
                if self._native_enabled:
                    try:
                        scan = win32_targets.collect(
                            (request.x, request.y), own_pid=os.getpid()
                        )
                        native = list(scan.targets)
                    except OSError:  # pragma: no cover - defensive
                        scan = WindowScan()
                        native = []

                targets = normalize_targets(
                    native,
                    screen=screen,
                    screen_label=request.screen_label,
                )
                stage = "win32" if self._native_enabled else "screen"
                self._emit(request, targets, stage, stop_event)

                if not (self._native_enabled and uia.available and scan.top_hwnd):
                    continue

                seed = win32_targets.select_uia_seed(scan)
                if not seed:
                    continue
                now = time.perf_counter()
                uia_list: list[UITarget] | None = None
                if last_uia is not None:
                    cached_seed, cached_targets, cached_at = last_uia
                    innermost = cached_targets[-1] if cached_targets else None
                    inside = bool(
                        innermost and innermost.rect.contains(request.x, request.y)
                    )
                    if (
                        cached_seed == seed
                        and inside
                        and now - cached_at < 0.30
                    ):
                        uia_list = (
                            _containing(cached_targets, request.x, request.y)
                            or None
                        )
                if (
                    uia_list is None
                    and last_uia is not None
                    and last_uia[0] == seed
                    and now - last_uia_at < 0.055
                ):
                    # New sample within one UIA frame: keep the previous
                    # element tree for the same HWND instead of queuing more
                    # cross-process work.
                    uia_list = (
                        _containing(last_uia[1], request.x, request.y)
                        or None
                    )
                if uia_list is None:
                    last_uia_at = time.perf_counter()
                    uia_list = uia.collect(
                        seed,
                        (request.x, request.y),
                        app=scan.app,
                        top_class=scan.top_class,
                    )
                    last_uia = (seed, uia_list, time.perf_counter())
                    if uia.error:
                        self._uia_hint = uia.error
                if uia_list is not None:
                    uia_list = _containing(uia_list, request.x, request.y)
                if not uia_list:
                    continue
                combined = normalize_targets(
                    [*native, *uia_list],
                    screen=screen,
                    screen_label=request.screen_label,
                )
                if _signature(combined) != _signature(targets):
                    self._emit(request, combined, "uia", stop_event)
        finally:
            uia.stop()


def target_stack_for(
    targets: list[UITarget], *, preserve: bool = False
) -> TargetStack:
    """Small helper kept for tests and callers that only need selection state."""
    stack = TargetStack()
    stack.update(targets, preserve=preserve)
    return stack
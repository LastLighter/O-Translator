"""Fixed-region pixel change monitor with debouncing and cooldowns."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import QThread, Signal

from client.services.screen_capture import (
    CaptureError,
    CaptureRegion,
    grab_bgr,
    open_capture,
    qimage_from_bgr,
)


@dataclass(frozen=True)
class MonitorSettings:
    threshold_percent: float = 5.0
    poll_interval_ms: int = 200
    debounce_ms: int = 400
    cooldown_ms: int = 2000
    noise_threshold: int = 12
    min_changed_pixels: int = 80
    static_percent: float = 0.8
    diff_max_side: int = 480

    def normalized(self) -> MonitorSettings:
        return replace(
            self,
            threshold_percent=min(40.0, max(0.5, float(self.threshold_percent))),
            poll_interval_ms=min(2000, max(100, int(self.poll_interval_ms))),
            debounce_ms=min(2000, max(100, int(self.debounce_ms))),
            cooldown_ms=min(10000, max(300, int(self.cooldown_ms))),
            noise_threshold=min(60, max(2, int(self.noise_threshold))),
            min_changed_pixels=max(1, int(self.min_changed_pixels)),
            static_percent=min(10.0, max(0.1, float(self.static_percent))),
            diff_max_side=min(1280, max(240, int(self.diff_max_side))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold_percent": self.threshold_percent,
            "poll_interval_ms": self.poll_interval_ms,
            "debounce_ms": self.debounce_ms,
            "cooldown_ms": self.cooldown_ms,
            "noise_threshold": self.noise_threshold,
            "min_changed_pixels": self.min_changed_pixels,
            "static_percent": self.static_percent,
            "diff_max_side": self.diff_max_side,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MonitorSettings:
        defaults = cls()
        return cls(
            threshold_percent=float(
                payload.get("threshold_percent", defaults.threshold_percent)
            ),
            poll_interval_ms=int(
                payload.get("poll_interval_ms", defaults.poll_interval_ms)
            ),
            debounce_ms=int(payload.get("debounce_ms", defaults.debounce_ms)),
            cooldown_ms=int(payload.get("cooldown_ms", defaults.cooldown_ms)),
            noise_threshold=int(
                payload.get("noise_threshold", defaults.noise_threshold)
            ),
            min_changed_pixels=int(
                payload.get("min_changed_pixels", defaults.min_changed_pixels)
            ),
            static_percent=float(
                payload.get("static_percent", defaults.static_percent)
            ),
            diff_max_side=int(payload.get("diff_max_side", defaults.diff_max_side)),
        ).normalized()


@dataclass(frozen=True)
class ChangeDecision:
    initialized: bool = False
    trigger: bool = False
    pending: bool = False
    change_ratio: float = 0.0
    frame_delta: float = 0.0
    changed_pixels: int = 0
    message: str = "等待画面变化"


class FrameAnalyzer:
    """Downscale frames and compute a noise-filtered changed-pixel ratio."""

    @staticmethod
    def prepare(frame_bgr: np.ndarray, max_side: int = 480) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        longest = max(height, width)
        if longest > max_side:
            scale = max_side / float(longest)
            gray = cv2.resize(
                gray,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return gray

    @staticmethod
    def difference(
        previous: np.ndarray,
        current: np.ndarray,
        *,
        noise_threshold: int = 12,
        min_changed_pixels: int = 80,
    ) -> tuple[float, int]:
        if previous.shape != current.shape:
            return 100.0, int(current.size)
        diff = cv2.absdiff(previous, current)
        changed = int(np.count_nonzero(diff > noise_threshold))
        if changed < min_changed_pixels:
            return 0.0, 0
        return changed / float(diff.size) * 100.0, changed


class ChangeDetector:
    """Pure state machine: baseline -> change -> settle -> debounce -> trigger."""

    def __init__(self, settings: MonitorSettings) -> None:
        self.settings = settings.normalized()
        self.baseline: np.ndarray | None = None
        self.previous: np.ndarray | None = None
        self.pending = False
        self.last_activity = 0.0
        self.last_trigger = float("-inf")

    def reset(self) -> None:
        self.baseline = None
        self.previous = None
        self.pending = False
        self.last_activity = 0.0
        self.last_trigger = float("-inf")

    def update(self, frame: np.ndarray, now: float) -> ChangeDecision:
        if self.baseline is None or self.previous is None:
            self.baseline = frame
            self.previous = frame
            return ChangeDecision(initialized=True, message="监控中 · 等待画面变化")

        frame_delta, _ = FrameAnalyzer.difference(
            self.previous,
            frame,
            noise_threshold=self.settings.noise_threshold,
            min_changed_pixels=self.settings.min_changed_pixels,
        )
        change_ratio, changed_pixels = FrameAnalyzer.difference(
            self.baseline,
            frame,
            noise_threshold=self.settings.noise_threshold,
            min_changed_pixels=self.settings.min_changed_pixels,
        )

        if self.pending:
            if frame_delta > self.settings.static_percent:
                self.last_activity = now
                self.previous = frame
                return ChangeDecision(
                    pending=True,
                    change_ratio=change_ratio,
                    frame_delta=frame_delta,
                    changed_pixels=changed_pixels,
                    message="画面仍在变化，等待稳定…",
                )
            if now - self.last_activity < self.settings.debounce_ms / 1000.0:
                self.previous = frame
                return ChangeDecision(
                    pending=True,
                    change_ratio=change_ratio,
                    frame_delta=frame_delta,
                    changed_pixels=changed_pixels,
                    message="检测到变化，等待画面稳定…",
                )
            if now - self.last_trigger >= self.settings.cooldown_ms / 1000.0:
                self.baseline = frame
                self.previous = frame
                self.pending = False
                self.last_trigger = now
                return ChangeDecision(
                    trigger=True,
                    pending=False,
                    change_ratio=change_ratio,
                    frame_delta=frame_delta,
                    changed_pixels=changed_pixels,
                    message=f"已触发翻译 · 变化 {change_ratio:.1f}%",
                )
            self.previous = frame
            remaining = self.settings.cooldown_ms / 1000.0 - (
                now - self.last_trigger
            )
            return ChangeDecision(
                pending=True,
                change_ratio=change_ratio,
                frame_delta=frame_delta,
                changed_pixels=changed_pixels,
                message=f"冷却中 · {remaining:.1f}s",
            )
        else:
            if change_ratio >= self.settings.threshold_percent:
                self.pending = True
                self.last_activity = now
                self.previous = frame
                return ChangeDecision(
                    pending=True,
                    change_ratio=change_ratio,
                    frame_delta=frame_delta,
                    changed_pixels=changed_pixels,
                    message=f"检测到变化 {change_ratio:.1f}%，等待稳定…",
                )

        self.previous = frame
        return ChangeDecision(
            change_ratio=change_ratio,
            frame_delta=frame_delta,
            changed_pixels=changed_pixels,
            message=f"监控中 · 变化 {change_ratio:.1f}%",
        )


class RegionMonitor(QThread):
    """Capture a fixed region periodically on a worker thread.

    The monitor only performs cheap grayscale diffing. Full RGB/QImage creation
    happens only when a translation is actually triggered.
    """

    triggered = Signal(object, float)
    stats = Signal(float, bool, str)
    status = Signal(str)
    error = Signal(str)
    finished_monitoring = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._region: CaptureRegion | None = None
        self._settings = MonitorSettings().normalized()
        self._settings_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._suspend_event = threading.Event()
        self._reset_event = threading.Event()
        self._paused = False
        self._exclusion_lock = threading.Lock()
        self._exclusion_rects: list[QtCore.QRect] = []

    @property
    def region(self) -> CaptureRegion | None:
        return self._region

    @property
    def settings(self) -> MonitorSettings:
        with self._settings_lock:
            return self._settings

    @property
    def paused(self) -> bool:
        return self._paused

    def start_monitoring(
        self, region: CaptureRegion, settings: MonitorSettings
    ) -> None:
        if self.isRunning():
            self.stop_monitoring()
        self._region = region
        with self._settings_lock:
            self._settings = settings.normalized()
        self._stop_event.clear()
        self._pause_event.clear()
        self._suspend_event.clear()
        self._reset_event.set()
        self._paused = False
        self.start()

    def update_settings(self, settings: MonitorSettings) -> None:
        with self._settings_lock:
            self._settings = settings.normalized()
        self._reset_event.set()

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        if paused:
            self._pause_event.set()
            self.status.emit("已暂停自动监控")
        else:
            self._pause_event.clear()
            self._reset_event.set()
            self.status.emit("自动监控已恢复")

    def set_exclusion(self, rect: QtCore.QRect | None) -> None:
        """Exclude the floating HUD from diffing to prevent feedback loops."""
        self.set_exclusions([rect] if rect is not None else [])

    def set_exclusions(self, rects: list[QtCore.QRect]) -> None:
        with self._exclusion_lock:
            self._exclusion_rects = [
                QtCore.QRect(rect)
                for rect in rects
                if rect is not None and not rect.isEmpty()
            ]

    def _exclusion_snapshot(self) -> list[QtCore.QRect]:
        with self._exclusion_lock:
            return [QtCore.QRect(rect) for rect in self._exclusion_rects]

    def suspend_capture(self, suspended: bool, reason: str = "") -> None:
        """Temporarily suspend frame grabs without changing the user pause state."""
        if suspended:
            self._suspend_event.set()
            if reason:
                self.status.emit(reason)
        else:
            self._suspend_event.clear()
            self._reset_event.set()

    def _apply_exclusion(
        self, gray: np.ndarray, region: CaptureRegion, logical_rect: QtCore.QRect
    ) -> None:
        physical = region.physical_subrect(logical_rect)
        if physical is None or physical.isEmpty():
            return
        scale_x = gray.shape[1] / max(1, region.physical_width)
        scale_y = gray.shape[0] / max(1, region.physical_height)
        x0 = max(0, int(physical.left() * scale_x) - 1)
        y0 = max(0, int(physical.top() * scale_y) - 1)
        x1 = min(gray.shape[1], int(physical.right() * scale_x) + 2)
        y1 = min(gray.shape[0], int(physical.bottom() * scale_y) + 2)
        if x1 > x0 and y1 > y0:
            gray[y0:y1, x0:x1] = 0

    def stop_monitoring(self) -> None:
        self._stop_event.set()
        self._pause_event.clear()
        if self.isRunning():
            self.wait(3000)

    def run(self) -> None:
        cv2.setNumThreads(1)
        region = self._region
        if region is None:
            self.error.emit("未选择监控区域。")
            self.finished_monitoring.emit()
            return
        detector: ChangeDetector | None = None
        settings_snapshot: MonitorSettings | None = None
        last_stats = 0.0
        failures = 0
        was_paused = False

        try:
            with open_capture() as capture:
                while not self._stop_event.is_set():
                    if self._pause_event.is_set():
                        if not was_paused:
                            was_paused = True
                            detector = None
                            self.status.emit("已暂停自动监控")
                        self.msleep(120)
                        continue
                    if self._suspend_event.is_set():
                        self.msleep(100)
                        continue
                    if was_paused:
                        was_paused = False
                        detector = None
                        self.status.emit("自动监控已恢复")

                    settings = self.settings
                    if self._reset_event.is_set():
                        self._reset_event.clear()
                        detector = None
                        settings_snapshot = None
                    if detector is None or settings != settings_snapshot:
                        detector = ChangeDetector(settings)
                        settings_snapshot = settings
                        self.status.emit("监控中 · 等待画面变化")

                    try:
                        frame = grab_bgr(region, capture)
                        failures = 0
                    except CaptureError as exc:
                        failures += 1
                        self.error.emit(str(exc))
                        if failures >= 5:
                            self.status.emit("监控已停止：连续截图失败")
                            break
                        self.msleep(1000)
                        continue

                    gray = FrameAnalyzer.prepare(frame, settings.diff_max_side)
                    for exclusion in self._exclusion_snapshot():
                        self._apply_exclusion(gray, region, exclusion)
                    now = time.monotonic()
                    decision = detector.update(gray, now)
                    if decision.trigger:
                        image = qimage_from_bgr(frame)
                        self.triggered.emit(image, decision.change_ratio)
                        self.status.emit(decision.message)
                    elif decision.pending and decision.message:
                        self.status.emit(decision.message)

                    if now - last_stats >= 0.5:
                        last_stats = now
                        self.stats.emit(
                            decision.change_ratio, decision.pending, decision.message
                        )
                    self.msleep(settings.poll_interval_ms)
        except Exception as exc:
            self.error.emit(f"监控线程异常：{exc}")
        finally:
            self.finished_monitoring.emit()
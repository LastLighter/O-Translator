"""DPI-aware screen enumeration and region capture.

Qt exposes screen geometry in device-independent pixels, while mss and Windows
capture APIs operate in physical pixels. This module pairs each QScreen with its
physical monitor, then converts logical selections using the measured
physical/logical scale factors. That keeps selections aligned on 125/150/200%
displays and on multi-monitor virtual desktops with negative origins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mss
import numpy as np
from PySide6 import QtCore, QtGui


class CaptureError(RuntimeError):
    pass


MSS_FACTORY = getattr(mss, "MSS", None) or mss.mss


def open_capture() -> Any:
    """Open an mss capture handle (new API with legacy fallback)."""
    return MSS_FACTORY()


@dataclass(frozen=True)
class ScreenTarget:
    name: str
    index: int
    logical_left: int
    logical_top: int
    logical_width: int
    logical_height: int
    physical_left: int
    physical_top: int
    physical_width: int
    physical_height: int

    @property
    def logical_rect(self) -> QtCore.QRect:
        return QtCore.QRect(
            self.logical_left,
            self.logical_top,
            self.logical_width,
            self.logical_height,
        )

    @property
    def physical_rect(self) -> QtCore.QRect:
        return QtCore.QRect(
            self.physical_left,
            self.physical_top,
            self.physical_width,
            self.physical_height,
        )

    @property
    def scale_x(self) -> float:
        return self.physical_width / max(1, self.logical_width)

    @property
    def scale_y(self) -> float:
        return self.physical_height / max(1, self.logical_height)

    @property
    def dpr(self) -> float:
        return (self.scale_x + self.scale_y) / 2.0

    def to_capture_region(self, logical_rect: QtCore.QRect) -> CaptureRegion:
        logical_rect = logical_rect.normalized().intersected(self.logical_rect)
        if logical_rect.width() < 1 or logical_rect.height() < 1:
            raise CaptureError("选区不在当前屏幕范围内。")
        local_left = logical_rect.left() - self.logical_left
        local_top = logical_rect.top() - self.logical_top
        physical_left = self.physical_left + round(local_left * self.scale_x)
        physical_top = self.physical_top + round(local_top * self.scale_y)
        physical_width = max(1, round(logical_rect.width() * self.scale_x))
        physical_height = max(1, round(logical_rect.height() * self.scale_y))
        return CaptureRegion(
            physical_left=physical_left,
            physical_top=physical_top,
            physical_width=physical_width,
            physical_height=physical_height,
            logical_left=logical_rect.left(),
            logical_top=logical_rect.top(),
            logical_width=logical_rect.width(),
            logical_height=logical_rect.height(),
            scale_x=self.scale_x,
            scale_y=self.scale_y,
            screen_name=self.name,
        )

    def physical_rect_to_logical(
        self,
        left: int,
        top: int,
        width: int,
        height: int,
        *,
        clip: bool = True,
    ) -> QtCore.QRect:
        """Map a physical virtual-desktop rectangle into Qt logical pixels.

        Win32 and UI Automation report physical pixels for a per-monitor
        DPI-aware process. The overlay draws in QScreen logical coordinates, so
        both the origin and the size must be divided by that screen's measured
        scale before painting or capturing.
        """
        if width <= 0 or height <= 0:
            return QtCore.QRect()
        logical_left = self.logical_left + round(
            (int(left) - self.physical_left) / self.scale_x
        )
        logical_top = self.logical_top + round(
            (int(top) - self.physical_top) / self.scale_y
        )
        logical_width = max(1, round(int(width) / self.scale_x))
        logical_height = max(1, round(int(height) / self.scale_y))
        rect = QtCore.QRect(
            logical_left, logical_top, logical_width, logical_height
        )
        return rect.intersected(self.logical_rect) if clip else rect


@dataclass(frozen=True)
class CaptureRegion:
    physical_left: int
    physical_top: int
    physical_width: int
    physical_height: int
    logical_left: int
    logical_top: int
    logical_width: int
    logical_height: int
    scale_x: float = 1.0
    scale_y: float = 1.0
    screen_name: str = ""

    @property
    def mss_monitor(self) -> dict[str, int]:
        return {
            "left": int(self.physical_left),
            "top": int(self.physical_top),
            "width": max(1, int(self.physical_width)),
            "height": max(1, int(self.physical_height)),
        }

    @property
    def logical_rect(self) -> QtCore.QRect:
        return QtCore.QRect(
            self.logical_left,
            self.logical_top,
            self.logical_width,
            self.logical_height,
        )

    @property
    def physical_rect(self) -> QtCore.QRect:
        return QtCore.QRect(
            self.physical_left,
            self.physical_top,
            self.physical_width,
            self.physical_height,
        )

    @property
    def logical_center(self) -> QtCore.QPoint:
        return self.logical_rect.center()

    @property
    def area(self) -> int:
        return max(0, self.physical_width) * max(0, self.physical_height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "physical_left": self.physical_left,
            "physical_top": self.physical_top,
            "physical_width": self.physical_width,
            "physical_height": self.physical_height,
            "logical_left": self.logical_left,
            "logical_top": self.logical_top,
            "logical_width": self.logical_width,
            "logical_height": self.logical_height,
            "scale_x": round(self.scale_x, 6),
            "scale_y": round(self.scale_y, 6),
            "screen_name": self.screen_name,
        }

    def physical_subrect(self, logical_rect: QtCore.QRect) -> QtCore.QRect | None:
        """Map a global logical rect into capture-local physical coordinates."""
        local = logical_rect.intersected(self.logical_rect)
        if local.width() <= 0 or local.height() <= 0:
            return None
        x = round((local.left() - self.logical_left) * self.scale_x)
        y = round((local.top() - self.logical_top) * self.scale_y)
        width = round(local.width() * self.scale_x)
        height = round(local.height() * self.scale_y)
        rect = QtCore.QRect(x, y, max(1, width), max(1, height))
        return rect.intersected(QtCore.QRect(0, 0, self.physical_width, self.physical_height))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CaptureRegion:
        required = (
            "physical_left",
            "physical_top",
            "physical_width",
            "physical_height",
            "logical_left",
            "logical_top",
            "logical_width",
            "logical_height",
        )
        for key in required:
            if key not in payload:
                raise ValueError(f"CaptureRegion is missing {key}")
        if int(payload["physical_width"]) < 1 or int(payload["physical_height"]) < 1:
            raise ValueError("CaptureRegion has invalid dimensions")
        return cls(
            physical_left=int(payload["physical_left"]),
            physical_top=int(payload["physical_top"]),
            physical_width=int(payload["physical_width"]),
            physical_height=int(payload["physical_height"]),
            logical_left=int(payload["logical_left"]),
            logical_top=int(payload["logical_top"]),
            logical_width=int(payload["logical_width"]),
            logical_height=int(payload["logical_height"]),
            scale_x=float(payload.get("scale_x", 1.0)),
            scale_y=float(payload.get("scale_y", 1.0)),
            screen_name=str(payload.get("screen_name", "")),
        )


@dataclass(frozen=True)
class CaptureResult:
    image: QtGui.QImage
    region: CaptureRegion
    source: str = "selection"


def virtual_desktop_rect(rects: list[QtCore.QRect]) -> QtCore.QRect:
    result = QtCore.QRect()
    for rect in rects:
        result = result.united(rect)
    return result


def _monitor_size(monitor: dict[str, Any]) -> tuple[int, int]:
    return int(monitor["width"]), int(monitor["height"])


def _pair_with_monitors(
    screens: list[Any],
    monitors: list[dict[str, Any]],
) -> list[ScreenTarget]:
    """Pair QScreens with physical monitors, preferring exact size matches."""
    available = list(range(len(monitors)))
    pairs: list[tuple[int, int]] = []

    for index, screen in enumerate(screens):
        geometry = screen.geometry()
        dpr = float(screen.devicePixelRatio() or 1.0)
        expected_w = max(1, round(geometry.width() * dpr))
        expected_h = max(1, round(geometry.height() * dpr))

        def score(
            candidate_index: int,
            *,
            expected_w: int = expected_w,
            expected_h: int = expected_h,
            geometry: QtCore.QRect = geometry,
            dpr: float = dpr,
        ) -> tuple[float, float]:
            monitor = monitors[candidate_index]
            width, height = _monitor_size(monitor)
            size_error = abs(width - expected_w) + abs(height - expected_h)
            expected_left = geometry.left() * dpr
            expected_top = geometry.top() * dpr
            origin_error = abs(float(monitor["left"]) - expected_left) + abs(
                float(monitor["top"]) - expected_top
            )
            return float(size_error), origin_error

        candidates = sorted(available, key=score) if available else []
        if candidates:
            best = candidates[0]
            pairs.append((index, best))
            available.remove(best)
        else:
            # No physical monitor information: synthesise one from DPR.
            synthetic = {
                "left": round(geometry.left() * dpr),
                "top": round(geometry.top() * dpr),
                "width": expected_w,
                "height": expected_h,
            }
            monitors.append(synthetic)
            pairs.append((index, len(monitors) - 1))

    targets: list[ScreenTarget] = []
    for screen_index, monitor_index in pairs:
        screen = screens[screen_index]
        geometry = screen.geometry()
        monitor = monitors[monitor_index]
        width, height = _monitor_size(monitor)
        targets.append(
            ScreenTarget(
                name=screen.name(),
                index=screen_index,
                logical_left=geometry.left(),
                logical_top=geometry.top(),
                logical_width=geometry.width(),
                logical_height=geometry.height(),
                physical_left=int(monitor["left"]),
                physical_top=int(monitor["top"]),
                physical_width=width,
                physical_height=height,
            )
        )
    return targets


def enumerate_screen_targets() -> list[ScreenTarget]:
    app = QtGui.QGuiApplication.instance()
    if app is None:
        raise CaptureError("QGuiApplication is not initialized.")
    screens = list(QtGui.QGuiApplication.screens())
    monitors: list[dict[str, Any]] = []
    try:
        with open_capture() as capture:
            monitors = [dict(item) for item in capture.monitors[1:]]
    except Exception:
        monitors = []
    return _pair_with_monitors(screens, monitors)


def virtual_desktop_geometry() -> QtCore.QRect:
    app = QtGui.QGuiApplication.instance()
    if app is None:
        raise CaptureError("QGuiApplication is not initialized.")
    return virtual_desktop_rect([screen.geometry() for screen in app.screens()])


def target_for_screen(screen: QtGui.QScreen) -> ScreenTarget:
    targets = enumerate_screen_targets()
    for target in targets:
        if target.logical_rect == screen.geometry():
            return target
    for target in targets:
        if target.name == screen.name():
            return target
    raise CaptureError("无法将当前屏幕映射到物理显示器。")


def target_for_logical_point(point: QtCore.QPoint) -> ScreenTarget | None:
    for target in enumerate_screen_targets():
        if target.logical_rect.contains(point):
            return target
    return None


def refresh_region(region: CaptureRegion) -> CaptureRegion:
    """Re-map a stored region after resolution/DPI/monitor changes."""
    targets = enumerate_screen_targets()
    center = region.logical_rect.center()
    best: ScreenTarget | None = None
    best_area = 0
    for target in targets:
        overlap = target.logical_rect.intersected(region.logical_rect)
        area = overlap.width() * overlap.height()
        if target.logical_rect.contains(center):
            best = target
            break
        if area > best_area:
            best = target
            best_area = area
    if best is None:
        raise CaptureError("保存的监控区域不在任何屏幕内。")
    return best.to_capture_region(region.logical_rect)


def grab_bgr(region: CaptureRegion, capture: Any | None = None) -> np.ndarray:
    try:
        if capture is None:
            with open_capture() as local:
                shot = local.grab(region.mss_monitor)
        else:
            shot = capture.grab(region.mss_monitor)
        frame = np.asarray(shot)
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise CaptureError("截图格式异常。")
        return np.ascontiguousarray(frame[:, :, :3])
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"屏幕捕获失败：{exc}") from exc


def qimage_from_bgr(frame: np.ndarray) -> QtGui.QImage:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise CaptureError("Invalid BGR frame.")
    height, width = frame.shape[:2]
    return QtGui.QImage(
        frame.data,
        width,
        height,
        int(frame.strides[0]),
        QtGui.QImage.Format.Format_BGR888,
    ).copy()


def capture_region_image(
    region: CaptureRegion, capture: Any | None = None
) -> QtGui.QImage:
    return qimage_from_bgr(grab_bgr(region, capture))


def capture_full_screen(target: ScreenTarget) -> CaptureResult:
    region = target.to_capture_region(target.logical_rect)
    return CaptureResult(
        image=capture_region_image(region), region=region, source="fullscreen"
    )


def capture_logical_rect(
    target: ScreenTarget, logical_rect: QtCore.QRect, source: str = "selection"
) -> CaptureResult:
    region = target.to_capture_region(logical_rect)
    return CaptureResult(
        image=capture_region_image(region), region=region, source=source
    )
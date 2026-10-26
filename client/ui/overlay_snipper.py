"""DPI-aware multi-monitor snipping overlay with Snipaste-style confirmation.

A separate pane is created for every QScreen. Each pane is sized in that
screen's own logical coordinate system, avoiding the mixed-DPI distortion that
a single window spanning multiple monitors can suffer. Selections are clamped
to the starting screen and mapped to physical pixels through that screen's
measured scale factors.

The overlay freezes the desktop once on open. Painting the frozen image (instead
of compositing a full-screen translucent layer on every mouse sample) keeps the
selection smooth even on 4K/high-refresh displays, and the frozen pixels are
also the source of the final crop.

Interaction follows Snipaste:

* the first frame highlights the window/control under the cursor captured when
  the hotkey was pressed;
* a left click locks that target, a left drag draws a manual selection, and
  releasing the drag locks the rectangle (it does not capture immediately);
* once locked, a floating toolbar offers translate/save/exit, dragging inside
  the rectangle moves it, and a right click steps back to free selection;
* another right click (or Esc from free selection) closes the overlay.
"""

from __future__ import annotations

import time
from contextlib import suppress
from datetime import datetime

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer

from client.config import CAPTURE_DIR, ensure_capture_dir
from client.services.screen_capture import (
    CaptureError,
    CaptureResult,
    ScreenTarget,
    capture_region_image,
    enumerate_screen_targets,
    open_capture,
)
from client.services.target_detector import TargetDetector, quick_targets
from client.services.ui_target import SCREEN, PhysicalRect, TargetStack, UITarget

_IDLE = "idle"
_DRAGGING = "dragging"
_LOCKED = "locked"
_MOVING = "moving"

_SOURCE_LABELS = {
    "selection": "选区",
    "window": "窗口",
    "fullscreen": "整屏",
}

_TOOLBAR_QSS = """
QFrame#SnipperToolbar {
    background: #14161d;
    border: 1px solid rgba(255, 255, 255, 42);
    border-radius: 8px;
}
QLabel#SnipperSize {
    color: #d8d5ce;
    font-family: "Cascadia Code", Consolas, monospace;
    font-size: 9px;
    padding: 0 3px;
}
QLabel#SnipperHint {
    color: #777b86;
    font-size: 9px;
    padding: 0 2px;
}
QPushButton#SnipperPrimary {
    background: #ff6b5e;
    color: #23100e;
    border: 0;
    border-radius: 5px;
    padding: 4px 11px;
    font-size: 9px;
    font-weight: 700;
}
QPushButton#SnipperPrimary:hover { background: #ff8176; }
QPushButton#SnipperPrimary:pressed { background: #e9564b; }
QPushButton#SnipperButton {
    background: rgba(255, 255, 255, 18);
    color: #d2cfc8;
    border: 1px solid rgba(255, 255, 255, 26);
    border-radius: 5px;
    padding: 4px 9px;
    font-size: 9px;
}
QPushButton#SnipperButton:hover {
    background: rgba(255, 255, 255, 32);
    color: #ffffff;
}
"""


def _crop_image(
    image: QtGui.QImage | None,
    target: ScreenTarget,
    region: object,
) -> QtGui.QImage | None:
    """Crop a frozen screen image using a physical capture region."""
    if image is None or image.isNull():
        return None
    left = int(getattr(region, "physical_left", 0)) - target.physical_left
    top = int(getattr(region, "physical_top", 0)) - target.physical_top
    width = int(getattr(region, "physical_width", 0))
    height = int(getattr(region, "physical_height", 0))
    if width <= 0 or height <= 0:
        return None
    local = QtCore.QRect(left, top, width, height).intersected(image.rect())
    if local.width() < 1 or local.height() < 1:
        return None
    return image.copy(local)


class _SelectionToolbar(QtWidgets.QFrame):
    """Small floating action bar shown after a selection is locked."""

    def __init__(self, pane: _OverlayPane) -> None:
        super().__init__(pane)
        self.pane = pane
        self.setObjectName("SnipperToolbar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet(_TOOLBAR_QSS)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(9, 4, 5, 4)
        layout.setSpacing(6)

        self.size_label = QtWidgets.QLabel("0 × 0")
        self.size_label.setObjectName("SnipperSize")
        self.hint_label = QtWidgets.QLabel("拖动移动 · 右键返回")
        self.hint_label.setObjectName("SnipperHint")
        self.save_btn = QtWidgets.QPushButton("保存")
        self.save_btn.setObjectName("SnipperButton")
        self.save_btn.setToolTip("保存选区截图")
        self.confirm_btn = QtWidgets.QPushButton("翻译")
        self.confirm_btn.setObjectName("SnipperPrimary")
        self.confirm_btn.setToolTip("确认选区并开始翻译（Enter）")
        self.cancel_btn = QtWidgets.QPushButton("退出")
        self.cancel_btn.setObjectName("SnipperButton")
        self.cancel_btn.setToolTip("退出截图（Esc）")

        for button in (self.save_btn, self.confirm_btn, self.cancel_btn):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setCursor(Qt.CursorShape.PointingHandCursor)

        layout.addWidget(self.size_label)
        layout.addWidget(self.hint_label)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.confirm_btn)
        layout.addWidget(self.cancel_btn)

        self.confirm_btn.clicked.connect(pane.confirm_capture)
        self.save_btn.clicked.connect(pane.save_selection)
        self.cancel_btn.clicked.connect(pane.cancel_overlay)
        self.adjustSize()

    def set_dimensions(self, width: int, height: int, source: str) -> None:
        source_label = _SOURCE_LABELS.get(source, "选区")
        self.size_label.setText(f"{width} × {height} · {source_label}")
        self.adjustSize()


class _OverlayPane(QtWidgets.QWidget):
    def __init__(
        self,
        controller: OverlaySnipper,
        target: ScreenTarget,
        index: int,
        backdrop: QtGui.QImage | None = None,
    ) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.controller = controller
        self.target = target
        self.index = index
        self._backdrop = backdrop
        if backdrop is not None and not backdrop.isNull():
            # Frozen screenshot: an opaque, fully painted window avoids the
            # per-pixel layered-window compositing cost while dragging.
            self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        else:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.setGeometry(target.logical_rect)

        self._state = _IDLE
        self._press_pos: QtCore.QPoint | None = None
        self._start: QtCore.QPoint | None = None
        self._current: QtCore.QPoint | None = None
        self._pressed_target: UITarget | None = None
        self._selection: QtCore.QRect | None = None
        self._selection_source = "selection"
        self._move_offset: QtCore.QPoint | None = None
        self._targets = TargetStack()
        self._detect_serial = -1
        self._target_signature: tuple[tuple[object, ...], ...] = ()

        self._last_request_at = 0.0
        self._last_request_point = QtCore.QPoint(-10_000, -10_000)

        self._toolbar = _SelectionToolbar(self)
        self._toolbar.hide()
        for widget in [self._toolbar, *self._toolbar.findChildren(QtWidgets.QWidget)]:
            widget.installEventFilter(self)

    # ------------------------------------------------------------- detection
    @property
    def target_stack(self) -> TargetStack:
        return self._targets

    @property
    def detect_serial(self) -> int:
        return self._detect_serial

    @property
    def backdrop(self) -> QtGui.QImage | None:
        return self._backdrop

    @property
    def state(self) -> str:
        return self._state

    def is_dragging(self) -> bool:
        return self._state == _DRAGGING

    def is_pressed(self) -> bool:
        return self._state in {_DRAGGING, _MOVING}

    def is_locked(self) -> bool:
        return self._state in {_LOCKED, _MOVING}

    def has_selection(self) -> bool:
        return self._selection is not None and not self._selection.isEmpty()

    @property
    def toolbar_visible(self) -> bool:
        return self._toolbar.isVisible()

    @property
    def toolbar(self) -> _SelectionToolbar:
        return self._toolbar

    def should_request_detection(
        self, global_point: QtCore.QPoint, *, force: bool = False
    ) -> bool:
        if force:
            self._last_request_at = time.perf_counter()
            self._last_request_point = QtCore.QPoint(global_point)
            return True
        now = time.perf_counter()
        moved = (global_point - self._last_request_point).manhattanLength()
        elapsed = now - self._last_request_at
        # The detector itself is capped at 60 Hz, but emitting a new serial for
        # every raw mouse event (some mice report 1000 Hz) churns the UI thread
        # and invalidates the target stack.  30 Hz is plenty for a highlight.
        if elapsed < 0.030:
            return False
        if moved < 2 and elapsed < 0.120:
            return False
        self._last_request_at = now
        self._last_request_point = QtCore.QPoint(global_point)
        return True

    def apply_targets(self, targets: list[UITarget], *, preserve: bool = False) -> None:
        if self._state != _IDLE:
            return
        signature = tuple(
            (target.rect, target.label, target.kind, target.source)
            for target in targets
        )
        if signature == self._target_signature:
            # Win32 and UIA often re-report the same chain while the pointer is
            # still; repainting and resetting the wheel-selected layer would
            # only add jitter.
            return
        before = self._visual_bounds()
        self._targets.update(targets, preserve=preserve)
        self._target_signature = signature
        after = self._visual_bounds()
        self._repaint_region(before.united(after))

    def refresh_target_visual(self, before: QtCore.QRect) -> None:
        self._repaint_region(before.united(self._visual_bounds()))

    def target_local_rect(self, target: UITarget) -> QtCore.QRect:
        screen_physical = PhysicalRect(
            self.target.physical_left,
            self.target.physical_top,
            self.target.physical_width,
            self.target.physical_height,
        )
        clipped = target.rect.intersection(screen_physical)
        if clipped.is_empty:
            return QtCore.QRect()
        global_rect = self.target.physical_rect_to_logical(
            clipped.left,
            clipped.top,
            clipped.width,
            clipped.height,
        )
        local = global_rect.translated(
            -self.target.logical_left, -self.target.logical_top
        )
        return local.intersected(self.rect())

    # ------------------------------------------------------------- geometry
    def selection_rect(self) -> QtCore.QRect:
        if self._state == _DRAGGING:
            if self._start is None or self._current is None:
                return QtCore.QRect()
            return QtCore.QRect(self._start, self._current).normalized()
        if self._state in {_LOCKED, _MOVING} and self._selection is not None:
            return QtCore.QRect(self._selection)
        return QtCore.QRect()

    def _visual_selection_rect(self) -> QtCore.QRect:
        if self._state in {_LOCKED, _MOVING}:
            return QtCore.QRect(self._selection) if self._selection else QtCore.QRect()
        if self._state == _DRAGGING:
            return self.selection_rect()
        if not self.controller.is_active(self.index):
            return QtCore.QRect()
        target = self._targets.current
        if target is None:
            return QtCore.QRect()
        return self.target_local_rect(target)

    def _selection_label(self, rect: QtCore.QRect) -> str:
        if self._state == _DRAGGING:
            return f"{rect.width()} × {rect.height()}"
        if self._state in {_LOCKED, _MOVING}:
            source = _SOURCE_LABELS.get(self._selection_source, "选区")
            return f"{rect.width()} × {rect.height()} · {source}"
        target = self._targets.current
        if target is None:
            return f"{rect.width()} × {rect.height()}"
        layer = (
            f"{self._targets.index + 1}/{self._targets.count} 层 · "
            if self._targets.count > 1
            else ""
        )
        return f"{layer}{target.display_label} · {rect.width()} × {rect.height()}"

    def _badge_geometry(
        self, rect: QtCore.QRect, text: str
    ) -> tuple[QtCore.QRect, str]:
        metrics = self.fontMetrics()
        max_width = max(140, min(520, self.width() - 20))
        elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight, max_width - 18)
        width = min(max_width, metrics.horizontalAdvance(elided) + 18)
        height = 23
        x = rect.left()
        y = rect.top() - height - 5
        if y < 4:
            y = rect.bottom() + 6
        if y + height > self.height() - 4:
            y = rect.top() + 6
        x = max(4, min(self.width() - width - 4, x))
        return QtCore.QRect(x, max(4, y), width, height), elided

    def _visual_bounds(self) -> QtCore.QRect:
        rect = self._visual_selection_rect()
        if rect.isEmpty():
            return QtCore.QRect()
        bounds = rect.adjusted(-4, -4, 4, 4)
        badge, _ = self._badge_geometry(rect, self._selection_label(rect))
        bounds = bounds.united(badge.adjusted(-2, -2, 2, 2))
        return bounds.intersected(self.rect())

    def _repaint_region(self, rect: QtCore.QRect) -> None:
        if rect.isNull() or rect.isEmpty():
            return
        self.update(rect.adjusted(-3, -3, 3, 3).intersected(self.rect()))

    # ------------------------------------------------------------------ paint
    def _physical_rect(self, local: QtCore.QRect) -> QtCore.QRect:
        if self._backdrop is None or self._backdrop.isNull():
            return QtCore.QRect()
        sx = self.target.scale_x
        sy = self.target.scale_y
        x = round(local.x() * sx)
        y = round(local.y() * sy)
        width = max(1, round(local.width() * sx))
        height = max(1, round(local.height() * sy))
        return QtCore.QRect(x, y, width, height).intersected(self._backdrop.rect())

    def _paint_image(self, painter: QtGui.QPainter, local: QtCore.QRect) -> None:
        """Blit the frozen image at 1:1 physical pixels.

        Drawing with ``drawImage(logical_rect, image, source_rect)`` would make
        Qt resample the whole pane on every fractional-DPI repaint.  Mapping the
        painter to physical pixels for this blit keeps adjacent dirty regions
        perfectly aligned and avoids soft/duplicated edges while dragging.
        """
        if self._backdrop is None or self._backdrop.isNull():
            return
        physical = self._physical_rect(local)
        if physical.isEmpty():
            return
        painter.save()
        painter.scale(1.0 / self.target.scale_x, 1.0 / self.target.scale_y)
        painter.drawImage(physical, self._backdrop, physical)
        painter.restore()

    def _paint_backdrop(self, painter: QtGui.QPainter, rect: QtCore.QRect) -> None:
        self._paint_image(painter, rect)
        painter.fillRect(rect, QtGui.QColor(7, 9, 12, 148))

    def _paint_bright(self, painter: QtGui.QPainter, rect: QtCore.QRect) -> None:
        if self._backdrop is not None and not self._backdrop.isNull():
            self._paint_image(painter, rect)
            return
        # No frozen frame (rare capture failure): cut a real hole through the
        # translucent pane so the live desktop shows through, like before.
        painter.save()
        painter.setCompositionMode(
            QtGui.QPainter.CompositionMode.CompositionMode_Clear
        )
        painter.fillRect(rect, Qt.GlobalColor.transparent)
        painter.restore()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        dirty = event.rect()
        painter.setClipRect(dirty)
        self._paint_backdrop(painter, dirty)

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 107, 94, 70), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self.rect().adjusted(1, 1, -1, -1))

        active = self.controller.is_active(self.index)
        if active:
            self._draw_screen_label(painter, dirty)

        selection = self._visual_selection_rect() if active else QtCore.QRect()
        if not selection.isEmpty():
            rect = selection.intersected(self.rect())
            if rect.intersects(dirty):
                self._paint_bright(painter, rect.intersected(dirty))
                self._draw_selection_chrome(painter, rect, dirty)

        if active and self._state == _IDLE:
            self._draw_instructions(painter, dirty)
            if (
                self.controller.native_targets_supported
                and self.controller.detector_notice
            ):
                self._draw_notice(painter, self.controller.detector_notice, dirty)

    def _draw_selection_chrome(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRect,
        dirty: QtCore.QRect,
    ) -> None:
        painter.setPen(QtGui.QPen(QtGui.QColor("#ff7b6e"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        if rect.width() >= 72 and rect.height() >= 54:
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 42), 1))
            for fraction in (1 / 3, 2 / 3):
                x = int(rect.left() + rect.width() * fraction)
                y = int(rect.top() + rect.height() * fraction)
                painter.drawLine(x, rect.top(), x, rect.bottom())
                painter.drawLine(rect.left(), y, rect.right(), y)

        badge, elided = self._badge_geometry(rect, self._selection_label(rect))
        if not badge.intersects(dirty):
            return
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(12, 14, 19, 232))
        painter.drawRoundedRect(badge, 5, 5)
        painter.setPen(QtGui.QColor("#ffb5ad"))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, elided)

    def _draw_screen_label(
        self, painter: QtGui.QPainter, dirty: QtCore.QRect
    ) -> None:
        text = f"屏幕 {self.index + 1} · {self.width()}×{self.height()}"
        font = QtGui.QFont(painter.font())
        font.setPointSize(8)
        font.setBold(True)
        painter.save()
        painter.setFont(font)
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 16
        badge = QtCore.QRect(self.width() - width - 10, 10, width, 22)
        if badge.intersects(dirty):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(15, 17, 23, 210))
            painter.drawRoundedRect(badge, 5, 5)
            painter.setPen(QtGui.QColor("#ffb5ad"))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    def _draw_notice(
        self, painter: QtGui.QPainter, text: str, dirty: QtCore.QRect
    ) -> None:
        font = QtGui.QFont(painter.font())
        font.setPointSize(8)
        font.setBold(False)
        painter.save()
        painter.setFont(font)
        metrics = painter.fontMetrics()
        max_width = max(160, self.width() - 40)
        elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight, max_width - 18)
        width = min(max_width, metrics.horizontalAdvance(elided) + 18)
        badge = QtCore.QRect(12, self.height() - 34, width, 23)
        if badge.intersects(dirty):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(34, 27, 16, 225))
            painter.drawRoundedRect(badge, 5, 5)
            painter.setPen(QtGui.QColor("#f4b95f"))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, elided)
        painter.restore()

    def _draw_instructions(
        self, painter: QtGui.QPainter, dirty: QtCore.QRect
    ) -> None:
        if not self.controller.native_targets_supported:
            title = "拖拽选择要翻译的区域"
            subtitle = "拖拽框选 · Enter 整屏 · 右键 / Esc 退出"
        elif self.controller.mode == "monitor":
            title = "选择要持续监控的窗口 / 区域"
            subtitle = (
                "移动鼠标识别 · 单击或空格锁定 · 滚轮切换层级 · "
                "拖拽框选 · Enter 整屏 · 右键 / Esc 退出"
            )
        else:
            title = "移动鼠标自动识别窗口 / 控件"
            subtitle = (
                "单击或空格锁定高亮元素 · 滚轮切换层级 · 拖拽框选 · "
                "Enter 整屏 · 右键 / Esc 退出"
            )
        card_width = min(560, self.width() - 40)
        card = QtCore.QRect(
            (self.width() - card_width) // 2,
            34,
            card_width,
            80,
        )
        if not card.intersects(dirty):
            return
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(16, 18, 24, 242))
        painter.drawRoundedRect(card, 14, 14)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 26), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(card, 14, 14)

        painter.save()
        font = QtGui.QFont(painter.font())
        font.setPointSize(13)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#f4f1e9"))
        painter.drawText(
            card.adjusted(10, 10, -10, -36),
            Qt.AlignmentFlag.AlignCenter,
            title,
        )
        font.setPointSize(9)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#8b8e98"))
        painter.drawText(
            card.adjusted(10, 42, -10, -8),
            Qt.AlignmentFlag.AlignCenter,
            subtitle,
        )
        painter.restore()

    # ------------------------------------------------------------- selection
    def _source_for_target(self, target: UITarget) -> str:
        if self.controller.mode == "monitor":
            return "selection"
        if target.kind == SCREEN:
            return "fullscreen"
        return "window"

    def _lock_rect(self, rect: QtCore.QRect, source: str) -> None:
        rect = rect.normalized().intersected(self.rect())
        if rect.width() < 2 or rect.height() < 2:
            return
        self._state = _LOCKED
        self._selection = rect
        self._selection_source = source
        self._press_pos = None
        self._start = None
        self._current = None
        self._pressed_target = None
        self._move_offset = None
        self._targets.update([])
        self._target_signature = ()
        self._toolbar.set_dimensions(rect.width(), rect.height(), source)
        self.controller.lock(self)
        self._place_toolbar()
        self._toolbar.show()
        self._toolbar.raise_()
        self.update()

    def _unlock(self) -> None:
        if self._state not in {_LOCKED, _MOVING}:
            return
        self._state = _IDLE
        self._selection = None
        self._selection_source = "selection"
        self._move_offset = None
        self._toolbar.hide()
        self._targets.update([])
        self._target_signature = ()
        if self.controller.locked_pane is self:
            self.controller.unlock(self)
        self.update()

    def _begin_drag(self, point: QtCore.QPoint) -> None:
        self._state = _DRAGGING
        self._start = QtCore.QPoint(self._press_pos or point)
        self._current = QtCore.QPoint(point)
        self._press_pos = None
        self._pressed_target = None
        self._targets.update([])
        self._target_signature = ()
        self._toolbar.hide()
        self.update()

    def _finish_drag(self, point: QtCore.QPoint) -> None:
        self._current = point
        rect = self.selection_rect()
        self._start = None
        self._current = None
        if rect.width() < 8 or rect.height() < 8:
            self._state = _IDLE
            self.update()
            self.controller.request_detection(
                self, QtGui.QCursor.pos(), force=True
            )
            return
        self._lock_rect(rect, "selection")

    def _reset_idle(self) -> None:
        if self._state != _IDLE:
            return
        self._press_pos = None
        self._pressed_target = None
        self._start = None
        self._current = None
        self.update()

    # ---------------------------------------------------------------- toolbar
    def _place_toolbar(self) -> None:
        selection = self._selection
        if selection is None:
            return
        toolbar = self._toolbar
        toolbar.adjustSize()
        width = max(1, toolbar.sizeHint().width())
        height = max(1, toolbar.sizeHint().height())
        toolbar.resize(width, height)
        margin = 8
        gap = 8
        spaces = {
            "below": self.height() - selection.bottom() - margin,
            "above": selection.top() - margin,
            "right": self.width() - selection.right() - margin,
            "left": selection.left() - margin,
        }
        candidates: list[tuple[str, int, int, int]] = []
        if spaces["below"] >= height + gap:
            candidates.append(
                (
                    "below",
                    selection.left(),
                    selection.bottom() + gap,
                    spaces["below"],
                )
            )
        if spaces["above"] >= height + gap:
            candidates.append(
                (
                    "above",
                    selection.left(),
                    selection.top() - gap - height,
                    spaces["above"],
                )
            )
        if spaces["right"] >= width + gap:
            candidates.append(
                (
                    "right",
                    selection.right() + gap,
                    selection.top(),
                    spaces["right"],
                )
            )
        if spaces["left"] >= width + gap:
            candidates.append(
                (
                    "left",
                    selection.left() - gap - width,
                    selection.top(),
                    spaces["left"],
                )
            )

        if candidates:
            # Largest free side wins; the explicit order resolves ties in
            # favour of below/right, which is where users expect the bar.
            preference = {"below": 0, "right": 1, "left": 2, "above": 3}
            _, x, y, _ = max(
                candidates,
                key=lambda item: (item[3], -preference[item[0]]),
            )
        else:
            # The selection touches (or nearly fills) the screen edges: keep
            # the toolbar inside, in a corner that still leaves the rectangle
            # mostly visible.
            if (
                selection.width() >= width + 32
                and selection.height() >= height + 32
            ):
                x = selection.right() - width - 8
                y = selection.bottom() - height - 8
            else:
                x = selection.left() + 8
                y = selection.top() + 8
        x = max(margin, min(self.width() - width - margin, x))
        y = max(margin, min(self.height() - height - margin, y))
        toolbar.move(x, y)

    # ------------------------------------------------------------------ input
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self.controller.back()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.controller.locked_pane is not None and not self.is_locked():
            return
        self.controller.activate(self.index)
        self.activateWindow()
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        with suppress(Exception):
            self.grabKeyboard()
        point = event.position().toPoint()
        point.setX(max(0, min(self.width() - 1, point.x())))
        point.setY(max(0, min(self.height() - 1, point.y())))

        if self._state == _LOCKED:
            if self._selection is not None and self._selection.contains(point):
                self._state = _MOVING
                self._move_offset = point - self._selection.topLeft()
                old_toolbar = self._toolbar.geometry()
                self._toolbar.hide()
                # With WA_OpaquePaintEvent the exposed area is not guaranteed to
                # be repainted by the platform; restore the frozen pixels here.
                self._repaint_region(old_toolbar)
                return
            # A press outside the locked rectangle starts a fresh selection.
            self._unlock()
            self._press_pos = point
            self._pressed_target = None
            return

        if self._state == _IDLE:
            self._press_pos = point
            self._pressed_target = self._targets.current

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        point = event.position().toPoint()
        point.setX(max(0, min(self.width() - 1, point.x())))
        point.setY(max(0, min(self.height() - 1, point.y())))

        if self._state == _MOVING:
            if self._selection is None or self._move_offset is None:
                return
            old_selection = QtCore.QRect(self._selection)
            old_toolbar = self._toolbar.geometry()
            new_left = point.x() - self._move_offset.x()
            new_top = point.y() - self._move_offset.y()
            new_left = max(0, min(self.width() - self._selection.width(), new_left))
            new_top = max(0, min(self.height() - self._selection.height(), new_top))
            self._selection.moveTo(new_left, new_top)
            self._place_toolbar()
            dirty = old_selection.united(self._selection)
            dirty = dirty.united(old_toolbar)
            dirty = dirty.united(self._toolbar.geometry())
            self._repaint_region(dirty)
            return

        if self._state == _DRAGGING:
            if self._current == point:
                return
            old = self.selection_rect()
            self._current = point
            new = self.selection_rect()
            self._repaint_region(old.united(new))
            return

        if self._press_pos is not None:
            if (point - self._press_pos).manhattanLength() >= 5:
                self._begin_drag(point)
            return

        self.controller.activate(self.index)
        self.controller.request_detection(
            self, event.globalPosition().toPoint()
        )

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = event.position().toPoint()
        point.setX(max(0, min(self.width() - 1, point.x())))
        point.setY(max(0, min(self.height() - 1, point.y())))

        if self._state == _MOVING:
            self._state = _LOCKED
            self._move_offset = None
            self._place_toolbar()
            self._toolbar.show()
            self._toolbar.raise_()
            self.update()
            return

        if self._state == _DRAGGING:
            self._finish_drag(point)
            return

        if self._press_pos is None:
            return
        target = self._pressed_target
        self._press_pos = None
        self._pressed_target = None
        if target is not None:
            rect = self.target_local_rect(target)
            if not rect.isEmpty():
                self._lock_rect(rect, self._source_for_target(target))
                return
        current = self._targets.current
        if current is not None:
            rect = self.target_local_rect(current)
            if not rect.isEmpty():
                self._lock_rect(rect, self._source_for_target(current))
                return
        self._reset_idle()
        self.controller.request_detection(
            self, event.globalPosition().toPoint(), force=True
        )

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self.controller.locked_pane is not None or self._state != _IDLE:
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        self.controller.activate(self.index)
        self.controller.cycle_targets(self, 1 if delta > 0 else -1)
        event.accept()

    def enterEvent(self, event: QtCore.QEvent) -> None:
        self.controller.activate(self.index)
        self.controller.request_detection(
            self, QtGui.QCursor.pos(), force=True
        )
        super().enterEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.controller.back()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.is_locked():
                self.confirm_capture()
            else:
                self._lock_rect(self.rect(), "fullscreen")
            return
        if event.key() == Qt.Key.Key_Space:
            if self.is_locked():
                self.confirm_capture()
            else:
                target = self._targets.current
                if target is not None:
                    rect = self.target_local_rect(target)
                    if not rect.isEmpty():
                        self._lock_rect(rect, self._source_for_target(target))
                        return
                self.controller.request_detection(
                    self, QtGui.QCursor.pos(), force=True
                )
            return
        if event.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
            if self.is_locked():
                return
            step = 1 if event.key() == Qt.Key.Key_Tab else -1
            self.controller.cycle_targets(self, step)
            return
        if (
            event.key() == Qt.Key.Key_S
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and self.is_locked()
        ):
            self.save_selection()
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if (
            event.type() == QtCore.QEvent.Type.MouseButtonPress
            and isinstance(event, QtGui.QMouseEvent)
            and event.button() == Qt.MouseButton.RightButton
        ):
            self.controller.back()
            return True
        return super().eventFilter(watched, event)

    # ---------------------------------------------------------------- actions
    def confirm_capture(self) -> None:
        if self._state not in {_LOCKED, _MOVING} or self._selection is None:
            return
        global_rect = self._selection.translated(
            self.target.logical_rect.topLeft()
        )
        self.controller.capture(
            self.target, global_rect, self._selection_source
        )

    def save_selection(self) -> None:
        if self._state not in {_LOCKED, _MOVING} or self._selection is None:
            return
        self.controller.save_selection(self, self._selection)

    def cancel_overlay(self) -> None:
        self.controller.cancel()


class OverlaySnipper(QtCore.QObject):
    captured = QtCore.Signal(object)
    cancelled = QtCore.Signal()
    failed = QtCore.Signal(str)
    saved = QtCore.Signal(str)

    def __init__(
        self,
        *,
        mode: str = "capture",
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.mode = mode
        self._panes: list[_OverlayPane] = []
        self._active_index: int | None = None
        self._locked_pane: _OverlayPane | None = None
        self._capturing = False
        self._open = False
        self._detector: TargetDetector | None = None
        self._serial = 0
        self._latest_serial: dict[int, int] = {}
        self._detector_notice = ""
        self._backdrops: dict[int, QtGui.QImage] = {}

    # ------------------------------------------------------------- properties
    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def panes(self) -> list[_OverlayPane]:
        return list(self._panes)

    @property
    def native_targets_supported(self) -> bool:
        return bool(self._detector and self._detector.native_enabled)

    @property
    def detector_notice(self) -> str:
        return self._detector_notice

    @property
    def locked_pane(self) -> _OverlayPane | None:
        return self._locked_pane

    # ---------------------------------------------------------------- lifecycle
    def open(self, cursor: QtCore.QPoint | None = None) -> None:
        if self._open:
            return
        try:
            targets = enumerate_screen_targets()
        except CaptureError as exc:
            self.failed.emit(str(exc))
            return
        self._open = True
        self._capturing = False
        self._serial = 0
        self._latest_serial.clear()
        self._locked_pane = None
        self._detector_notice = ""
        self._active_index = None
        self._backdrops = self._capture_backdrops(targets)
        self._ensure_detector()

        cursor_point = QtCore.QPoint(cursor) if cursor is not None else QtGui.QCursor.pos()
        focus_index = 0
        for index, target in enumerate(targets):
            if target.logical_rect.contains(cursor_point):
                focus_index = index
                break

        for index, target in enumerate(targets):
            pane = _OverlayPane(
                self,
                target,
                index,
                self._backdrops.get(target.index),
            )
            self._panes.append(pane)

        if self._panes:
            prime_pane = self._panes[focus_index]
            self._prime_pane(prime_pane, cursor_point)

        for pane in self._panes:
            pane.show()
            pane.raise_()

        self.activate(focus_index)
        if self._panes:
            pane = self._panes[focus_index]
            pane.activateWindow()
            pane.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
            with suppress(Exception):
                pane.grabKeyboard()
            self.request_detection(pane, cursor_point, force=True)

    def _capture_backdrops(
        self, targets: list[ScreenTarget]
    ) -> dict[int, QtGui.QImage]:
        app = QtGui.QGuiApplication.instance()
        platform = (
            QtGui.QGuiApplication.platformName().lower() if app is not None else ""
        )
        if platform in {"offscreen", "minimal"}:
            return {}
        images: dict[int, QtGui.QImage] = {}
        try:
            with open_capture() as capture:
                for target in targets:
                    try:
                        region = target.to_capture_region(target.logical_rect)
                        image = capture_region_image(region, capture)
                    except Exception:
                        continue
                    if not image.isNull():
                        images[target.index] = image
        except Exception:
            return {}
        return images

    def _prime_pane(
        self, pane: _OverlayPane, global_point: QtCore.QPoint
    ) -> None:
        x, y = self._physical_point(pane.target, global_point)
        try:
            targets = quick_targets(
                (x, y),
                screen=PhysicalRect(
                    pane.target.physical_left,
                    pane.target.physical_top,
                    pane.target.physical_width,
                    pane.target.physical_height,
                ),
                screen_label=f"屏幕 {pane.index + 1}",
            )
        except Exception:
            # The asynchronous detector is the fallback; a broken Win32 probe
            # must not prevent the overlay from opening.
            return
        pane.apply_targets(targets)

    def _ensure_detector(self) -> None:
        if self._detector is None:
            detector = TargetDetector(self)
            detector.detected.connect(self._on_detected)
            detector.notice.connect(self._on_detector_notice)
            self._detector = detector
        self._detector.start()

    def _on_detector_notice(self, message: str) -> None:
        self._detector_notice = message
        if self._active_index is not None and 0 <= self._active_index < len(
            self._panes
        ):
            self._panes[self._active_index].update()

    # --------------------------------------------------------------- detection
    def request_detection(
        self,
        pane: _OverlayPane,
        global_point: QtCore.QPoint,
        *,
        force: bool = False,
    ) -> None:
        if not self._open or self._capturing or self._locked_pane is not None:
            return
        if pane not in self._panes or pane.state != _IDLE:
            return
        if not pane.should_request_detection(global_point, force=force):
            return
        self._request_detection_at(pane, global_point)

    def _physical_point(
        self, target: ScreenTarget, global_point: QtCore.QPoint
    ) -> tuple[int, int]:
        x = target.physical_left + round(
            (global_point.x() - target.logical_left) * target.scale_x
        )
        y = target.physical_top + round(
            (global_point.y() - target.logical_top) * target.scale_y
        )
        x = max(
            target.physical_left,
            min(target.physical_left + target.physical_width - 1, x),
        )
        y = max(
            target.physical_top,
            min(target.physical_top + target.physical_height - 1, y),
        )
        return x, y

    def _request_detection_at(
        self, pane: _OverlayPane, global_point: QtCore.QPoint
    ) -> None:
        if self._detector is None:
            return
        target = pane.target
        x, y = self._physical_point(target, global_point)
        self._serial += 1
        serial = self._serial
        pane._detect_serial = serial
        self._latest_serial[pane.index] = serial
        self._detector.request(
            serial=serial,
            pane_index=pane.index,
            x=x,
            y=y,
            screen_rect=PhysicalRect(
                target.physical_left,
                target.physical_top,
                target.physical_width,
                target.physical_height,
            ),
            screen_label=f"屏幕 {pane.index + 1}",
        )

    def _on_detected(
        self,
        serial: int,
        pane_index: int,
        targets: object,
        stage: str,
    ) -> None:
        if not self._open or self._capturing or self._locked_pane is not None:
            return
        if pane_index < 0 or pane_index >= len(self._panes):
            return
        if serial != self._latest_serial.get(pane_index):
            return
        pane = self._panes[pane_index]
        if pane.state != _IDLE:
            return
        if isinstance(targets, list):
            pane.apply_targets(targets, preserve=stage == "uia")

    # ---------------------------------------------------------------- selection
    def activate(self, index: int) -> None:
        if self._locked_pane is not None and index != self._locked_pane.index:
            return
        if not (0 <= index < len(self._panes)):
            return
        changed = index != self._active_index
        self._active_index = index
        if not changed:
            # Mouse move handlers call activate() constantly; repainting every
            # pane here would make the whole desktop flash at pointer rate.
            return
        # Keyboard focus must follow the pointer across monitors, otherwise
        # Space/Esc would keep acting on the pane where the overlay opened.
        for pane in self._panes:
            if pane.index == index:
                pane.activateWindow()
                pane.setFocus(Qt.FocusReason.MouseFocusReason)
                with suppress(Exception):
                    pane.grabKeyboard()
            else:
                with suppress(Exception):
                    pane.releaseKeyboard()
            # Screen label, instruction card and target highlight are all
            # active-pane-only, so both panes need one repaint on a switch.
            pane.update()

    def is_active(self, index: int) -> bool:
        return index == self._active_index

    def lock(self, pane: _OverlayPane) -> None:
        if pane not in self._panes:
            return
        self._locked_pane = pane
        self._active_index = pane.index
        pane.activateWindow()
        pane.setFocus(Qt.FocusReason.MouseFocusReason)
        with suppress(Exception):
            pane.grabKeyboard()
        for other in self._panes:
            if other is not pane:
                with suppress(Exception):
                    other.releaseKeyboard()
                other.update()
        pane.update()

    def unlock(self, pane: _OverlayPane) -> None:
        if self._locked_pane is not pane:
            return
        self._locked_pane = None
        cursor = QtGui.QCursor.pos()
        for other in self._panes:
            other.update()
        # Focus and detect on the screen that actually contains the pointer; a
        # right click may have been aimed at a different monitor than the one
        # that owned the locked selection.
        target_pane = next(
            (
                other
                for other in self._panes
                if other.target.logical_rect.contains(cursor)
            ),
            pane,
        )
        self.activate(target_pane.index)
        self.request_detection(target_pane, cursor, force=True)

    def back(self) -> None:
        if self._locked_pane is not None:
            self._locked_pane._unlock()
            return
        self.cancel()

    def cycle_targets(self, pane: _OverlayPane, step: int) -> bool:
        if pane not in self._panes or self._locked_pane is not None:
            return False
        if pane.state != _IDLE:
            return False
        self.activate(pane.index)
        before = pane._visual_bounds()
        target = pane.target_stack.cycle(step)
        pane.refresh_target_visual(before)
        return target is not None

    # ----------------------------------------------------------------- capture
    def capture_target(self, pane: _OverlayPane, target: UITarget) -> None:
        if self._capturing or pane not in self._panes:
            return
        screen_physical = PhysicalRect(
            pane.target.physical_left,
            pane.target.physical_top,
            pane.target.physical_width,
            pane.target.physical_height,
        )
        clipped = target.rect.intersection(screen_physical)
        if clipped.is_empty:
            self.failed.emit("识别到的区域不在当前屏幕内。")
            return
        logical_rect = pane.target.physical_rect_to_logical(
            clipped.left,
            clipped.top,
            clipped.width,
            clipped.height,
        )
        if logical_rect.width() < 1 or logical_rect.height() < 1:
            self.failed.emit("识别到的区域尺寸无效。")
            return
        source = self._source_for_target(target)
        self.capture(pane.target, logical_rect, source)

    def _source_for_target(self, target: UITarget) -> str:
        if self.mode == "monitor":
            return "selection"
        if target.kind == SCREEN:
            return "fullscreen"
        return "window"

    def capture(
        self,
        target: ScreenTarget,
        logical_rect: QtCore.QRect,
        source: str,
    ) -> None:
        if self._capturing:
            return
        self._capturing = True
        frozen = self._backdrops.get(target.index)
        self._hide_panes()
        if frozen is not None and not frozen.isNull():
            self._grab_frozen(target, logical_rect, source, frozen)
            return
        QTimer.singleShot(
            110, lambda: self._grab(target, logical_rect, source)
        )

    def _grab_frozen(
        self,
        target: ScreenTarget,
        logical_rect: QtCore.QRect,
        source: str,
        frozen: QtGui.QImage,
    ) -> None:
        try:
            region = target.to_capture_region(logical_rect)
            image = _crop_image(frozen, target, region)
            if image is None or image.isNull():
                self._grab(target, logical_rect, source)
                return
            self.captured.emit(
                CaptureResult(image=image, region=region, source=source)
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.close()

    def _grab(
        self, target: ScreenTarget, logical_rect: QtCore.QRect, source: str
    ) -> None:
        try:
            region = target.to_capture_region(logical_rect)
            image = capture_region_image(region)
            if image.isNull():
                raise CaptureError("捕获结果为空。")
            self.captured.emit(
                CaptureResult(image=image, region=region, source=source)
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.close()

    def save_selection(
        self, pane: _OverlayPane, logical_rect: QtCore.QRect
    ) -> None:
        if self._capturing or pane not in self._panes:
            return
        try:
            region = pane.target.to_capture_region(
                logical_rect.translated(pane.target.logical_rect.topLeft())
            )
        except CaptureError as exc:
            QtWidgets.QMessageBox.warning(pane, "保存截图", str(exc))
            return
        frozen = self._backdrops.get(pane.target.index)
        image = _crop_image(frozen, pane.target, region)
        hidden = False
        if image is None:
            # Frozen frame unavailable: hide the overlay before grabbing the
            # live desktop, otherwise the saved image would contain the overlay.
            hidden = True
            self._hide_panes()
            try:
                image = capture_region_image(region)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(pane, "保存截图", str(exc))
                self.cancel()
                return
        try:
            ensure_capture_dir()
            default = (
                CAPTURE_DIR / f"O-Translator-{datetime.now():%Y%m%d-%H%M%S}.png"
            )
        except OSError:
            default = (
                QtCore.QDir.homePath()
                + f"/O-Translator-{datetime.now():%Y%m%d-%H%M%S}.png"
            )
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            pane,
            "保存截图",
            str(default),
            "PNG 图片 (*.png)",
        )
        if not path:
            if hidden:
                self.cancel()
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        if image is None or image.isNull() or not image.save(path, "PNG"):
            QtWidgets.QMessageBox.warning(pane, "保存截图", "保存失败。")
            if hidden:
                self.cancel()
            return
        self.saved.emit(path)
        self.close()

    # ------------------------------------------------------------------ close
    def cancel(self) -> None:
        self.cancelled.emit()
        self.close()

    def close(self) -> None:
        if not self._open and not self._panes:
            return
        self._open = False
        self._capturing = False
        if self._detector is not None:
            self._detector.stop()
        self._latest_serial.clear()
        self._backdrops.clear()
        self._hide_panes()
        for pane in self._panes:
            pane.deleteLater()
        self._panes.clear()
        self._active_index = None
        self._locked_pane = None

    def _hide_panes(self) -> None:
        for pane in self._panes:
            with suppress(Exception):
                pane.releaseKeyboard()
            with suppress(Exception):
                pane.toolbar.hide()
            pane.hide()
        QtWidgets.QApplication.processEvents()


__all__ = ["OverlaySnipper"]
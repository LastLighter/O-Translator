"""Always-on-top floating translation HUD.

The HUD deliberately stays out of the way. Screenshots result in an exact,
interactive overlay on the captured region; automatic region monitoring leaves
the HUD click-through by default so the underlying game/application keeps
receiving the pointer.

There is no hover title bar. The translated image itself is the drag handle, a
right click opens save/close, and a soft blue glow marks the outer edge so the
frameless overlay stays recognizable on top of busy content. The window keeps
transparent margins for that glow; the image still covers the captured region
pixel-for-pixel.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import sys
import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from client.config import APP_NAME
from client.services.screen_capture import CaptureRegion
from client.ui.styles import APP_STYLE


class FloatingTranslationWindow(QtWidgets.QWidget):
    closed = QtCore.Signal()
    click_through_changed = QtCore.Signal(bool)
    geometry_changed = QtCore.Signal(object)

    # Transparent room reserved around the frame for the blue glow. Keeping the
    # glow outside the frame means it never covers the captured pixels.
    SHADOW_MARGIN = 16
    SHADOW_SPREAD = 14
    FRAME_RADIUS = 12
    GLOW_COLOR = QtGui.QColor(64, 140, 255)

    WM_NCHITTEST = 0x0084
    HTCLIENT = 0
    HTTRANSPARENT = -1

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setStyleSheet(APP_STYLE)
        margin = self.SHADOW_MARGIN
        self.setMinimumSize(40 + margin * 2, 24 + margin * 2)

        self._image: QtGui.QImage | None = None
        self._region: CaptureRegion | None = None
        self._source = "capture"
        self._fade_anim: QtCore.QPropertyAnimation | None = None
        self._drag_offset: QtCore.QPoint | None = None
        # Click-through is opt-in per result: automatic monitoring enables it,
        # area captures stay interactive.
        self._click_through = False
        self._user_moved = False
        self._applying_geometry = False
        self._avoid_region = False
        self._aligned = False
        self._glow: QtGui.QPixmap | None = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(margin, margin, margin, margin)
        outer.setSpacing(0)
        self._frame = QtWidgets.QFrame()
        self._frame.setObjectName("FloatingFrame")
        outer.addWidget(self._frame)
        self._layout = QtWidgets.QVBoxLayout(self._frame)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        self._image_label = QtWidgets.QLabel()
        self._image_label.setObjectName("FloatingImage")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumHeight(1)
        self._image_label.setMouseTracking(True)
        self._image_label.setCursor(Qt.CursorShape.OpenHandCursor)
        self._image_label.setToolTip("拖动移动图片 · 右键保存或关闭")
        self._image_label.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._image_label.customContextMenuRequested.connect(
            self._show_context_menu
        )
        self._image_label.installEventFilter(self)
        self._layout.addWidget(self._image_label, 1)

        self._grip = QtWidgets.QSizeGrip(self)
        self._grip.setFixedSize(15, 15)
        self._grip.raise_()
        self.setWindowOpacity(1.0)

    # ------------------------------------------------------------------ public
    def show_result(
        self,
        image: QtGui.QImage,
        region: CaptureRegion | None = None,
        *,
        source: str = "capture",
        align: bool = True,
        avoid_region: bool = False,
        activate: bool = False,
        click_through: bool | None = None,
        fade: bool = False,
    ) -> None:
        if image.isNull():
            return
        first_show = not self.isVisible()
        changed_region = region != self._region
        self._image = image
        self._region = region
        self._source = source
        self._avoid_region = avoid_region
        # Click-through belongs to automatic monitoring. A manually captured
        # area is a finished overlay the user drags, saves, or closes, so it must
        # accept the pointer unless the tray toggle explicitly says otherwise.
        if click_through is None:
            click_through = source in {"monitor", "fullscreen"}
        self.set_click_through(bool(click_through))
        # Snapping is the default behavior. A new capture or region always
        # re-aligns; there is no manual snap button.
        if region is not None and (first_show or changed_region or align):
            self._user_moved = False
            self._apply_geometry(region, allow_align=align)
        elif first_show:
            self._apply_geometry(region, allow_align=align)
        self._refresh_pixmap()
        self.show()
        self.raise_()
        # Layout assigns the image label its final size on the next event-loop
        # turn; refresh again so the pixmap is scaled to that exact size instead
        # of keeping the pre-layout 640x480 default. The glow depends on the
        # frame geometry for the same reason.
        QtCore.QTimer.singleShot(0, self._after_layout)
        if activate and not self._click_through:
            self.activateWindow()
            self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        if fade:
            self._fade_pixmap()
        self.geometry_changed.emit(self.frameGeometry())

    def update_image(self, image: QtGui.QImage, *, fade: bool = True) -> None:
        """Swap the displayed translation without recomputing geometry.

        Used for the progressive second pass: the pass-1 preview is already on
        screen, and LaMa's result replaces it with a short cross-fade.
        """
        if image.isNull():
            return
        self._image = image
        self._refresh_pixmap()
        QtCore.QTimer.singleShot(0, self._refresh_pixmap)
        if fade:
            self._fade_pixmap()
        self.geometry_changed.emit(self.frameGeometry())

    def _after_layout(self) -> None:
        self._refresh_pixmap()
        self._rebuild_glow()

    def _fade_pixmap(self) -> None:
        """Fade the image label in after a preview -> final swap."""
        if self._image_label.pixmap() is None or self._image_label.pixmap().isNull():
            return
        if self._fade_anim is not None:
            self._fade_anim.stop()
            self._fade_anim = None
        effect = QtWidgets.QGraphicsOpacityEffect(self._image_label)
        effect.setOpacity(0.35)
        self._image_label.setGraphicsEffect(effect)
        animation = QtCore.QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(180)
        animation.setStartValue(0.35)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        animation.finished.connect(self._finish_fade)
        animation.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
        self._fade_anim = animation

    def _finish_fade(self) -> None:
        self._image_label.setGraphicsEffect(None)
        self._fade_anim = None

    @property
    def click_through(self) -> bool:
        return self._click_through

    @property
    def aligned(self) -> bool:
        return self._aligned

    def content_geometry(self) -> QtCore.QRect:
        """Global rectangle covered by the captured/translated image."""
        return self._frame.geometry().translated(self.geometry().topLeft())

    def set_click_through(self, enabled: bool) -> None:
        """Enable or disable smart click-through.

        When enabled, the whole HUD is hit-test transparent so the pointer
        reaches the application below. Disabling it makes the image draggable
        and the right-click save/close menu reachable.
        """
        enabled = bool(enabled)
        if enabled == self._click_through:
            return
        self._click_through = enabled
        self._update_interaction_cursor()
        self.click_through_changed.emit(enabled)

    def _update_interaction_cursor(self) -> None:
        if self._drag_offset is not None:
            cursor = Qt.CursorShape.ClosedHandCursor
        elif self._click_through:
            cursor = Qt.CursorShape.ArrowCursor
        else:
            cursor = Qt.CursorShape.OpenHandCursor
        self._image_label.setCursor(cursor)

    def snap_to_region(self) -> None:
        """Re-align to the last capture region (kept for API compatibility)."""
        self._user_moved = False
        self._apply_geometry(self._region, allow_align=True)
        self._refresh_pixmap()
        self.raise_()
        self.geometry_changed.emit(self.frameGeometry())

    # ------------------------------------------------------------------ layout
    def _apply_geometry(
        self, region: CaptureRegion | None, *, allow_align: bool = True
    ) -> None:
        self._applying_geometry = True
        try:
            screen = None
            if region is not None:
                screen = QtGui.QGuiApplication.screenAt(region.logical_center)
            if screen is None:
                screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())
            if screen is None:
                screen = QtGui.QGuiApplication.primaryScreen()
            available = screen.availableGeometry()
            screen_rect = screen.geometry()
            aligned = (
                allow_align
                and region is not None
                and not self._avoid_region
                and self._can_align(region, screen_rect)
            )
            self._set_aligned(aligned)
            if aligned and region is not None:
                self._align_geometry(region)
            else:
                self._near_geometry(
                    region, available, avoid_region=self._avoid_region
                )
        finally:
            self._applying_geometry = False

    def _can_align(self, region: CaptureRegion, screen_rect: QtCore.QRect) -> bool:
        width = region.logical_width
        height = region.logical_height
        # Small capture regions (game chat boxes, single dialogue bubbles) must
        # align too; the old 300x140 threshold silently turned them into a
        # detached panel, which is what made the overlay feel misplaced.
        return (
            width >= 8
            and height >= 8
            and region.logical_left >= screen_rect.left() - 2
            and region.logical_top >= screen_rect.top() - 2
            and region.logical_left + width <= screen_rect.right() + 3
            and region.logical_top + height <= screen_rect.bottom() + 3
        )

    def _align_geometry(self, region: CaptureRegion) -> None:
        # Exact pixel overlay: the transparent shadow margin sits outside the
        # capture region, so the image still lands on the original pixels.
        margin = self.SHADOW_MARGIN
        self.setGeometry(
            region.logical_left - margin,
            region.logical_top - margin,
            region.logical_width + margin * 2,
            region.logical_height + margin * 2,
        )

    def _near_geometry(
        self,
        region: CaptureRegion | None,
        available: QtCore.QRect,
        *,
        avoid_region: bool = False,
    ) -> None:
        margin = self.SHADOW_MARGIN
        max_content_width = max(160, available.width() - margin * 2 - 16)
        max_content_height = max(120, available.height() - margin * 2 - 16)
        target_width = min(760, max(420, int(available.width() * 0.55)))
        target_width = min(target_width, max_content_width)
        text_width = max(1, self._image.width() if self._image else target_width)
        text_height = max(1, self._image.height() if self._image else 400)
        content_height = int(target_width * text_height / text_width)
        content_height = min(content_height, max_content_height)
        width = target_width
        height = content_height
        if region is not None:
            right_x = region.logical_left + region.logical_width + 14
            left_x = region.logical_left - width - 14
            above_y = region.logical_top - height - 12
            below_y = region.logical_top + region.logical_height + 12
            candidates = [
                (right_x, region.logical_top),
                (left_x, region.logical_top),
                (region.logical_left, above_y),
                (region.logical_left, below_y),
            ]
            chosen: tuple[int, int] | None = None
            for x, y in candidates:
                if (
                    x >= available.left() + margin
                    and y >= available.top() + margin
                    and x + width <= available.right() - margin
                    and y + height <= available.bottom() - margin
                ):
                    chosen = (x, y)
                    break
            if chosen is None:
                if avoid_region:
                    x = available.right() - width - margin - 10
                    y = available.bottom() - height - margin - 10
                else:
                    x = region.logical_left
                    y = region.logical_top
            else:
                x, y = chosen
        else:
            x = available.center().x() - width // 2
            y = available.center().y() - height // 2
        x = max(
            available.left() + margin + 8,
            min(x, available.right() - margin - width - 8),
        )
        y = max(
            available.top() + margin + 8,
            min(y, available.bottom() - margin - height - 8),
        )
        self.setGeometry(
            x - margin,
            y - margin,
            width + margin * 2,
            height + margin * 2,
        )

    def _set_aligned(self, aligned: bool) -> None:
        if aligned == self._aligned:
            self._update_frame_style()
            return
        self._aligned = aligned
        self._grip.setVisible(not aligned)
        self._update_frame_style()
        self._rebuild_glow()

    def _update_frame_style(self) -> None:
        self._frame.setProperty("aligned", "true" if self._aligned else "false")
        self._frame.style().unpolish(self._frame)
        self._frame.style().polish(self._frame)
        self._image_label.setProperty(
            "aligned", "true" if self._aligned else "false"
        )
        self._image_label.style().unpolish(self._image_label)
        self._image_label.style().polish(self._image_label)

    # ------------------------------------------------------------------ paint
    def _refresh_pixmap(self) -> None:
        if self._image is None:
            return
        target = self._image_label.size()
        if target.width() < 2 or target.height() < 2:
            return
        pixmap = QtGui.QPixmap.fromImage(self._image)
        mode = (
            Qt.AspectRatioMode.IgnoreAspectRatio
            if self._aligned
            else Qt.AspectRatioMode.KeepAspectRatio
        )
        # Keep the physical pixel density on HiDPI screens: scaling to logical
        # size and letting Qt upscale again made the overlay look softer than
        # the capture (and contributed to the "resolution changed" perception).
        ratio = max(1.0, float(self.devicePixelRatioF()))
        physical = QtCore.QSize(
            max(1, round(target.width() * ratio)),
            max(1, round(target.height() * ratio)),
        )
        scaled = pixmap.scaled(
            physical,
            mode,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(ratio)
        self._image_label.setPixmap(scaled)

    def _rebuild_glow(self) -> None:
        """Cache the blue outer glow for the current frame geometry.

        The glow is painted by this top-level widget, not by a Qt graphics
        effect, so it never blurs the translated image itself.
        """
        width = self.width()
        height = self.height()
        if width < 2 or height < 2:
            self._glow = None
            return
        ratio = max(1.0, float(self.devicePixelRatioF()))
        pixmap = QtGui.QPixmap(
            max(1, round(width * ratio)),
            max(1, round(height * ratio)),
        )
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.scale(ratio, ratio)
            frame = QtCore.QRectF(self._frame.geometry())
            radius = 0.0 if self._aligned else float(self.FRAME_RADIUS)
            # Draw wide, faint rings first and tighter, brighter rings last. The
            # frame/image covers the inner half of each pen, leaving a soft blue
            # halo outside the border.
            spread = self.SHADOW_SPREAD
            for step in range(spread, 0, -1):
                progress = step / spread
                alpha = int(2 + 22 * (1.0 - progress) ** 2)
                color = QtGui.QColor(self.GLOW_COLOR)
                color.setAlpha(alpha)
                painter.setPen(QtGui.QPen(color, float(step * 2)))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(frame, radius, radius)
        finally:
            painter.end()
        pixmap.setDevicePixelRatio(ratio)
        self._glow = pixmap

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if self._glow is not None:
            painter = QtGui.QPainter(self)
            painter.drawPixmap(0, 0, self._glow)
        super().paintEvent(event)

    # ------------------------------------------------------------------ events
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        frame = self._frame.geometry()
        self._grip.move(
            frame.x() + frame.width() - self._grip.width() - 3,
            frame.y() + frame.height() - self._grip.height() - 3,
        )
        if not self._aligned:
            self._grip.raise_()
        self._refresh_pixmap()
        self._rebuild_glow()
        if self.isVisible() and not self._applying_geometry:
            self._user_moved = True
            self._set_aligned(False)
            self.geometry_changed.emit(self.frameGeometry())

    def moveEvent(self, event: QtGui.QMoveEvent) -> None:
        super().moveEvent(event)
        if self.isVisible() and not self._applying_geometry:
            self._user_moved = True
            self._set_aligned(False)
            self.geometry_changed.emit(self.frameGeometry())

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self._image_label:
            if (
                event.type() == QtCore.QEvent.Type.MouseButtonPress
                and isinstance(event, QtGui.QMouseEvent)
                and event.button() == Qt.MouseButton.LeftButton
                and not self._click_through
            ):
                self._drag_offset = (
                    event.globalPosition().toPoint()
                    - self.frameGeometry().topLeft()
                )
                self._update_interaction_cursor()
                self.raise_()
                self.activateWindow()
                return True
            if (
                event.type() == QtCore.QEvent.Type.MouseMove
                and isinstance(event, QtGui.QMouseEvent)
                and self._drag_offset is not None
                and event.buttons() & Qt.MouseButton.LeftButton
            ):
                self.move(event.globalPosition().toPoint() - self._drag_offset)
                return True
            if (
                event.type() == QtCore.QEvent.Type.MouseButtonRelease
                and isinstance(event, QtGui.QMouseEvent)
                and event.button() == Qt.MouseButton.LeftButton
                and self._drag_offset is not None
            ):
                self._drag_offset = None
                self._update_interaction_cursor()
                return True
        return super().eventFilter(watched, event)

    # ----------------------------------------------------------- native input
    def nativeEvent(self, event_type, message):
        """Return HTTRANSPARENT for the HUD while click-through is active."""
        if sys.platform == "win32" and self._click_through:
            try:
                msg = ctypes.wintypes.MSG.from_address(int(message))
            except (TypeError, ValueError, OSError):
                msg = None
            if msg is not None and msg.message == self.WM_NCHITTEST:
                x = ctypes.c_short(msg.lParam & 0xFFFF).value
                y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
                local = self.mapFromGlobal(QtCore.QPoint(x, y))
                if self.rect().contains(local):
                    if self._grip.isVisible() and self._grip.geometry().contains(
                        local
                    ):
                        return True, self.HTCLIENT
                    return True, self.HTTRANSPARENT
        return super().nativeEvent(event_type, message)

    # ------------------------------------------------------------------ actions
    def _show_context_menu(self, position: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        save_action = menu.addAction("保存图片")
        save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save)
        menu.addSeparator()
        close_action = menu.addAction("关闭浮窗")
        close_action.setShortcut(QtGui.QKeySequence(Qt.Key.Key_Escape))
        close_action.triggered.connect(self.close)
        menu.exec(self._image_label.mapToGlobal(position))

    def _save(self) -> None:
        if self._image is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存翻译图片",
            str(
                Path.home()
                / f"o-translator-{int(time.time())}.png"
            ),
            "PNG 图片 (*.png)",
        )
        if path and not self._image.save(path, "PNG"):
            QtWidgets.QMessageBox.warning(self, APP_NAME, "保存失败。")

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and not self._click_through:
            self.close()
            return
        if event.matches(QtGui.QKeySequence.StandardKey.Save):
            self._save()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._drag_offset = None
        if self._fade_anim is not None:
            self._fade_anim.stop()
            self._fade_anim = None
        self._image_label.setGraphicsEffect(None)
        self._glow = None
        if self._click_through:
            self._click_through = False
            self.click_through_changed.emit(False)
        self.closed.emit()
        super().closeEvent(event)

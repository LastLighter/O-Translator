from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from client.services.screen_capture import CaptureRegion
from client.services.ui_target import (
    CONTROL,
    ELEMENT,
    SCREEN,
    WINDOW,
    PhysicalRect,
    UITarget,
    screen_target,
)
from client.ui.floating_window import FloatingTranslationWindow
from client.ui.overlay_snipper import OverlaySnipper


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def _region() -> CaptureRegion:
    return CaptureRegion(
        physical_left=300,
        physical_top=300,
        physical_width=400,
        physical_height=200,
        logical_left=150,
        logical_top=150,
        logical_width=400,
        logical_height=200,
        scale_x=1.0,
        scale_y=1.0,
        screen_name="OFFSCREEN",
    )


def test_floating_window_aligns_to_region_and_defaults_to_click_through(qapp):
    image = QtGui.QImage(400, 200, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("#20242c"))
    window = FloatingTranslationWindow()
    changes: list[bool] = []
    window.click_through_changed.connect(changes.append)
    # Region monitoring owns the screen area and stays click-through; a manual
    # area capture must remain interactive.
    assert window.click_through is False
    window.show_result(image, _region(), source="monitor")
    qapp.processEvents()
    assert window.isVisible()
    assert window.click_through is True
    assert changes == [True]
    # The captured image still lands exactly on the captured region; the
    # transparent shadow margin lives outside it.
    margin = window.SHADOW_MARGIN
    assert window.content_geometry() == QtCore.QRect(150, 150, 400, 200)
    assert window.width() == 400 + margin * 2
    assert window.height() == 200 + margin * 2
    assert (window.x(), window.y()) == (150 - margin, 150 - margin)
    assert window.findChild(QtWidgets.QFrame, "FloatingBar") is None
    window.set_click_through(False)
    qapp.processEvents()
    assert window.click_through is False
    assert changes == [True, False]
    window.set_click_through(True)
    qapp.processEvents()
    assert window.click_through is True
    assert changes == [True, False, True]
    window.close()


def test_area_capture_is_interactive_and_aligns_small_regions(qapp):
    window = FloatingTranslationWindow()
    image = QtGui.QImage(120, 80, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("#20242c"))
    region = CaptureRegion(
        physical_left=150,
        physical_top=150,
        physical_width=120,
        physical_height=80,
        logical_left=150,
        logical_top=150,
        logical_width=120,
        logical_height=80,
        screen_name="OFFSCREEN",
    )
    window.show_result(image, region, source="selection")
    qapp.processEvents()
    assert window.click_through is False
    assert window.aligned is True
    # A tiny capture must not be inflated to the old 320x150 minimum; the
    # aligned image itself is exactly 120x80 on the region.
    margin = window.SHADOW_MARGIN
    assert window.content_geometry() == QtCore.QRect(150, 150, 120, 80)
    assert (window.width(), window.height()) == (120 + margin * 2, 80 + margin * 2)
    # The aligned image fills the content area exactly (no letterbox bars).
    pixmap = window._image_label.pixmap()
    assert pixmap is not None
    assert pixmap.size() == window._image_label.size()
    assert window._image_label.size() == QtCore.QSize(120, 80)
    window.close()


def test_image_drag_moves_window_and_detaches_alignment(qapp):
    window = FloatingTranslationWindow()
    image = QtGui.QImage(400, 200, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("#20242c"))
    window.show_result(image, _region(), source="selection")
    qapp.processEvents()
    assert window.aligned is True
    label = window._image_label
    local = label.rect().center()
    start = label.mapToGlobal(local)
    before = window.pos()

    press = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonPress,
        QtCore.QPointF(local),
        QtCore.QPointF(start),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    qapp.sendEvent(label, press)
    moved = start + QtCore.QPoint(40, 30)
    move = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        QtCore.QPointF(local + QtCore.QPoint(40, 30)),
        QtCore.QPointF(moved),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    qapp.sendEvent(label, move)
    qapp.processEvents()
    assert window.pos() == before + QtCore.QPoint(40, 30)
    assert window.aligned is False

    release = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonRelease,
        QtCore.QPointF(local + QtCore.QPoint(40, 30)),
        QtCore.QPointF(moved),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    qapp.sendEvent(label, release)
    assert window._drag_offset is None
    window.close()


def test_blue_glow_is_painted_outside_the_frame(qapp):
    window = FloatingTranslationWindow()
    image = QtGui.QImage(400, 200, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("#20242c"))
    window.show_result(image, _region(), source="selection")
    qapp.processEvents()
    assert window._glow is not None
    assert not window._glow.isNull()
    rendered = window.grab().toImage()
    # Halfway between the window edge and the frame: glow only, no image.
    x = window.SHADOW_MARGIN // 2
    color = rendered.pixelColor(x, rendered.height() // 2)
    assert color.alpha() > 0
    assert color.blue() > color.red()
    window.close()


def test_update_image_swaps_pixmap_for_progressive_result(qapp):
    window = FloatingTranslationWindow()
    preview = QtGui.QImage(400, 200, QtGui.QImage.Format.Format_RGB32)
    preview.fill(QtGui.QColor("#20242c"))
    window.show_result(preview, _region(), source="monitor")
    qapp.processEvents()
    final = QtGui.QImage(400, 200, QtGui.QImage.Format.Format_RGB32)
    final.fill(QtGui.QColor("#2c2024"))
    window.update_image(final, fade=True)
    qapp.processEvents()
    assert window._image is final
    pixmap = window._image_label.pixmap()
    assert pixmap is not None
    assert pixmap.size() == window._image_label.size()
    window.close()


def test_overlay_geometry_covers_virtual_desktop(qapp):
    overlay = OverlaySnipper(mode="monitor")
    overlay.open()
    qapp.processEvents()
    assert overlay.is_open
    assert len(overlay.panes) == len(QtGui.QGuiApplication.screens())
    for pane, screen in zip(
        overlay.panes, QtGui.QGuiApplication.screens(), strict=False
    ):
        assert pane.geometry() == screen.geometry()
    assert overlay.panes[0].selection_rect().isEmpty()
    overlay.close()
    assert not overlay.is_open


def _open_offline_overlay(qapp) -> OverlaySnipper:
    overlay = OverlaySnipper(mode="capture")
    overlay.open()
    QTest.qWait(60)
    qapp.processEvents()
    if overlay._detector is not None:
        overlay._detector.stop()
    qapp.processEvents()
    return overlay


def _fake_stack(pane) -> list[UITarget]:
    target = pane.target
    scale_x = max(1.0, target.scale_x)
    scale_y = max(1.0, target.scale_y)
    inner = UITarget(
        PhysicalRect(
            target.physical_left + round(40 * scale_x),
            target.physical_top + round(30 * scale_y),
            max(12, round(120 * scale_x)),
            max(12, round(60 * scale_y)),
        ),
        "按钮",
        CONTROL,
        "win32",
    )
    outer = UITarget(
        PhysicalRect(
            target.physical_left,
            target.physical_top,
            target.physical_width,
            target.physical_height,
        ),
        "窗口",
        WINDOW,
        "win32",
    )
    screen = screen_target(
        PhysicalRect(
            target.physical_left,
            target.physical_top,
            target.physical_width,
            target.physical_height,
        ),
        "屏幕 1",
    )
    pane.apply_targets([inner, outer, screen])
    return [inner, outer, screen]


def test_overlay_wheel_cycles_target_hierarchy(qapp):
    overlay = _open_offline_overlay(qapp)
    try:
        pane = overlay.panes[0]
        inner, outer, screen = _fake_stack(pane)
        assert pane.target_stack.current is inner
        assert overlay.cycle_targets(pane, 1) is True
        assert pane.target_stack.current is outer
        overlay.cycle_targets(pane, 1)
        assert pane.target_stack.current is screen
        overlay.cycle_targets(pane, 1)
        assert pane.target_stack.current is inner
    finally:
        overlay.close()


def test_uia_enrichment_moves_default_selection_to_innermost(qapp):
    overlay = _open_offline_overlay(qapp)
    try:
        pane = overlay.panes[0]
        window = UITarget(
            PhysicalRect(
                pane.target.physical_left,
                pane.target.physical_top,
                pane.target.physical_width,
                pane.target.physical_height,
            ),
            "窗口",
            WINDOW,
            "win32",
        )
        page = UITarget(
            PhysicalRect(
                window.rect.left + 30,
                window.rect.top + 30,
                max(40, window.rect.width - 60),
                max(40, window.rect.height - 60),
            ),
            "网页内容",
            CONTROL,
            "win32",
        )
        # First Win32 frame: only the coarse page is known.
        pane.apply_targets([page, window])
        assert pane.target_stack.current is page
        # UIA then resolves the control under the pointer; the default must
        # follow it instead of staying on the coarse Win32 page.
        text = UITarget(
            PhysicalRect(
                page.rect.left + 20,
                page.rect.top + 20,
                max(16, page.rect.width // 3),
                max(16, page.rect.height // 4),
            ),
            "段落",
            ELEMENT,
            "uia",
        )
        pane.apply_targets([text, page, window], preserve=True)
        assert pane.target_stack.current is text
        # An explicit wheel choice of an outer layer survives UIA enrichment.
        overlay.cycle_targets(pane, 1)
        assert pane.target_stack.current is page
        deeper = UITarget(
            PhysicalRect(
                text.rect.left + 4,
                text.rect.top + 4,
                max(12, text.rect.width // 2),
                max(12, text.rect.height // 2),
            ),
            "文字",
            ELEMENT,
            "uia",
        )
        pane.apply_targets([deeper, text, page, window], preserve=True)
        assert pane.target_stack.current is page
    finally:
        overlay.close()


def test_click_locks_highlighted_target_then_toolbar_confirms(qapp, monkeypatch):
    overlay = _open_offline_overlay(qapp)
    try:
        pane = overlay.panes[0]
        _fake_stack(pane)
        calls: list[tuple[QtCore.QRect, str]] = []
        monkeypatch.setattr(
            overlay,
            "capture",
            lambda target, rect, source: calls.append((rect, source)),
        )
        point = pane.target_local_rect(pane.target_stack.current).center()
        QTest.mouseClick(
            pane, QtCore.Qt.MouseButton.LeftButton, pos=point
        )
        qapp.processEvents()
        # The click selects the highlighted layer and shows the confirmation
        # toolbar; it must not start a translation job on the spot.
        assert calls == []
        assert pane.is_locked()
        assert pane.toolbar_visible
        assert not pane.selection_rect().isEmpty()
        pane.confirm_capture()
        qapp.processEvents()
        assert len(calls) == 1
        rect, source = calls[0]
        assert source == "window"
        assert rect.width() > 0 and rect.height() > 0

        # Right click steps back to free selection, then Space locks the same
        # layer again (keyboard equivalent of a click).
        QTest.mouseClick(
            pane, QtCore.Qt.MouseButton.RightButton, pos=point
        )
        qapp.processEvents()
        assert not pane.is_locked()
        assert not pane.toolbar_visible
        _fake_stack(pane)
        QTest.keyClick(pane, QtCore.Qt.Key.Key_Space)
        qapp.processEvents()
        assert pane.is_locked()
        pane.confirm_capture()
        qapp.processEvents()
        assert len(calls) == 2
        assert calls[1][1] == "window"
    finally:
        overlay.close()


def test_drag_overrides_auto_target_with_manual_selection(qapp, monkeypatch):
    overlay = _open_offline_overlay(qapp)
    try:
        pane = overlay.panes[0]
        _fake_stack(pane)
        calls: list[tuple[QtCore.QRect, str]] = []
        monkeypatch.setattr(
            overlay,
            "capture",
            lambda target, rect, source: calls.append((rect, source)),
        )
        QTest.mousePress(
            pane, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(100, 100)
        )
        QTest.mouseMove(pane, QtCore.QPoint(240, 200))
        QTest.mouseRelease(
            pane, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(240, 200)
        )
        qapp.processEvents()
        # Releasing the drag now locks the rectangle and reveals the toolbar.
        assert calls == []
        assert pane.is_locked()
        assert pane.toolbar_visible
        rect = pane.selection_rect()
        assert (rect.width(), rect.height()) == (141, 101)
        assert pane.target_stack.current is None
        pane.confirm_capture()
        qapp.processEvents()
        assert len(calls) == 1
        captured, source = calls[0]
        assert source == "selection"
        assert (captured.width(), captured.height()) == (141, 101)
    finally:
        overlay.close()


def test_click_on_screen_layer_uses_fullscreen_source(qapp, monkeypatch):
    overlay = _open_offline_overlay(qapp)
    try:
        pane = overlay.panes[0]
        _fake_stack(pane)
        overlay.cycle_targets(pane, 2)
        assert pane.target_stack.current.kind == SCREEN
        calls: list[tuple[QtCore.QRect, str]] = []
        monkeypatch.setattr(
            overlay,
            "capture",
            lambda target, rect, source: calls.append((rect, source)),
        )
        QTest.mouseClick(
            pane,
            QtCore.Qt.MouseButton.LeftButton,
            pos=pane.target_local_rect(pane.target_stack.current).center(),
        )
        qapp.processEvents()
        assert calls == []
        pane.confirm_capture()
        qapp.processEvents()
        assert calls and calls[0][1] == "fullscreen"
    finally:
        overlay.close()


def test_right_click_steps_back_then_closes(qapp):
    overlay = _open_offline_overlay(qapp)
    try:
        pane = overlay.panes[0]
        _fake_stack(pane)
        point = pane.target_local_rect(pane.target_stack.current).center()
        QTest.mouseClick(pane, QtCore.Qt.MouseButton.LeftButton, pos=point)
        qapp.processEvents()
        assert pane.is_locked()
        QTest.mouseClick(pane, QtCore.Qt.MouseButton.RightButton, pos=point)
        qapp.processEvents()
        assert not pane.is_locked()
        assert overlay.is_open
        assert not pane.toolbar_visible
        QTest.mouseClick(pane, QtCore.Qt.MouseButton.RightButton, pos=point)
        qapp.processEvents()
        assert not overlay.is_open
    finally:
        overlay.close()


def test_locked_selection_can_be_moved_with_left_drag(qapp):
    overlay = _open_offline_overlay(qapp)
    try:
        pane = overlay.panes[0]
        _fake_stack(pane)
        QTest.mousePress(
            pane, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(100, 100)
        )
        QTest.mouseMove(pane, QtCore.QPoint(240, 200))
        QTest.mouseRelease(
            pane, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(240, 200)
        )
        qapp.processEvents()
        before = pane.selection_rect()
        assert pane.is_locked()
        QTest.mousePress(
            pane, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(150, 150)
        )
        QTest.mouseMove(pane, QtCore.QPoint(180, 170))
        QTest.mouseRelease(
            pane, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(180, 170)
        )
        qapp.processEvents()
        after = pane.selection_rect()
        assert (after.x(), after.y()) == (before.x() + 30, before.y() + 20)
        assert (after.width(), after.height()) == (
            before.width(),
            before.height(),
        )
        assert pane.is_locked()
    finally:
        overlay.close()


def test_toolbar_sits_outside_selection_and_inside_for_fullscreen(qapp):
    overlay = _open_offline_overlay(qapp)
    try:
        pane = overlay.panes[0]
        _fake_stack(pane)
        point = pane.target_local_rect(pane.target_stack.current).center()
        QTest.mouseClick(pane, QtCore.Qt.MouseButton.LeftButton, pos=point)
        qapp.processEvents()
        selection = pane.selection_rect()
        toolbar = pane.toolbar.geometry()
        assert pane.toolbar_visible
        # The target is near the top-left, so there is plenty of room below and
        # the action bar stays outside the selected pixels.
        assert not toolbar.intersects(selection)
        assert toolbar.top() > selection.bottom()

        QTest.keyClick(pane, QtCore.Qt.Key.Key_Escape)
        qapp.processEvents()
        assert not pane.is_locked()
        QTest.keyClick(pane, QtCore.Qt.Key.Key_Return)
        qapp.processEvents()
        assert pane.is_locked()
        # A full-screen selection has no outside room: the toolbar moves inside
        # the selection but remains fully on-screen.
        assert pane.rect().contains(pane.toolbar.geometry())
        assert pane.selection_rect().contains(pane.toolbar.geometry())
    finally:
        overlay.close()


def test_save_selection_writes_crop_and_emits_saved(
    qapp, monkeypatch, tmp_path
):
    overlay = _open_offline_overlay(qapp)
    try:
        pane = overlay.panes[0]
        backdrop = QtGui.QImage(
            pane.target.physical_width,
            pane.target.physical_height,
            QtGui.QImage.Format.Format_RGB32,
        )
        backdrop.fill(QtGui.QColor("#334455"))
        overlay._backdrops[pane.target.index] = backdrop
        pane._lock_rect(QtCore.QRect(40, 30, 120, 60), "selection")
        qapp.processEvents()

        target_path = tmp_path / "saved-selection.png"
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(target_path), "PNG 图片 (*.png)"),
        )
        saved: list[str] = []
        overlay.saved.connect(saved.append)
        pane.save_selection()
        qapp.processEvents()
        assert target_path.is_file()
        assert saved == [str(target_path)]
        assert not overlay.is_open
    finally:
        overlay.close()


def test_open_primes_first_highlight_from_hotkey_cursor(
    qapp, monkeypatch
):
    from client.ui import overlay_snipper as overlay_module

    overlay = OverlaySnipper(mode="capture")
    recorded: list[tuple[int, int]] = []

    def fake_quick(point, *, screen, screen_label):
        recorded.append((point[0], point[1]))
        return [screen_target(screen, screen_label)]

    monkeypatch.setattr(overlay_module, "quick_targets", fake_quick)
    try:
        overlay.open(QtCore.QPoint(123, 77))
        qapp.processEvents()
        pane = overlay.panes[0]
        expected = overlay._physical_point(pane.target, QtCore.QPoint(123, 77))
        assert recorded == [expected]
        assert pane.target_stack.current is not None
        assert pane.target_stack.current.kind == SCREEN
    finally:
        if overlay._detector is not None:
            overlay._detector.stop()
        overlay.close()


def test_image_canvas_scales_without_error(qapp):
    from client.ui.widgets import ImageCanvas

    canvas = ImageCanvas()
    canvas.resize(640, 480)
    image = QtGui.QImage(300, 200, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("white"))
    canvas.set_image(image)
    qapp.processEvents()
    assert canvas.pixmap() is not None
    assert not canvas.pixmap().isNull()


def test_default_hotkey_is_f2():
    from client.config import DEFAULT_HOTKEY
    from client.services.hotkey import GlobalHotkey

    assert DEFAULT_HOTKEY == "f2"
    hotkey = GlobalHotkey("f2")
    assert hotkey._vk == 0x71
    assert hotkey.display == "F2"
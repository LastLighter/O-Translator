from __future__ import annotations

from types import SimpleNamespace

from PySide6 import QtCore

from client.services.screen_capture import (
    CaptureRegion,
    ScreenTarget,
    _pair_with_monitors,
    virtual_desktop_rect,
)


def test_screen_target_maps_fractional_dpi_selection():
    target = ScreenTarget(
        name="DISPLAY1",
        index=0,
        logical_left=1920,
        logical_top=0,
        logical_width=2560,
        logical_height=1440,
        physical_left=2400,
        physical_top=0,
        physical_width=3200,
        physical_height=1800,
    )
    region = target.to_capture_region(QtCore.QRect(2020, 100, 400, 300))
    assert region.physical_left == 2400 + 125
    assert region.physical_top == 125
    assert region.physical_width == 500
    assert region.physical_height == 375
    assert region.logical_left == 2020
    assert region.logical_width == 400
    assert region.scale_x == 1.25
    assert region.scale_y == 1.25


def test_screen_target_handles_negative_virtual_origin():
    target = ScreenTarget(
        name="DISPLAY2",
        index=1,
        logical_left=-1920,
        logical_top=0,
        logical_width=1920,
        logical_height=1080,
        physical_left=-1920,
        physical_top=0,
        physical_width=1920,
        physical_height=1080,
    )
    region = target.to_capture_region(QtCore.QRect(-1800, 50, 400, 200))
    assert region.physical_left == -1800
    assert region.physical_top == 50
    assert region.physical_width == 400
    assert region.physical_height == 200


def test_screen_target_clamps_to_screen_bounds():
    target = ScreenTarget(
        name="DISPLAY1",
        index=0,
        logical_left=0,
        logical_top=0,
        logical_width=800,
        logical_height=600,
        physical_left=0,
        physical_top=0,
        physical_width=1600,
        physical_height=1200,
    )
    region = target.to_capture_region(QtCore.QRect(700, 500, 300, 300))
    assert region.logical_rect == QtCore.QRect(700, 500, 100, 100)
    assert region.physical_rect == QtCore.QRect(1400, 1000, 200, 200)


def test_physical_target_rect_maps_back_to_logical_pixels():
    target = ScreenTarget(
        name="DISPLAY1",
        index=0,
        logical_left=1920,
        logical_top=0,
        logical_width=2560,
        logical_height=1440,
        physical_left=2400,
        physical_top=0,
        physical_width=3200,
        physical_height=1800,
    )
    rect = target.physical_rect_to_logical(2525, 250, 500, 250)
    assert rect == QtCore.QRect(2020, 200, 400, 200)
    # Outside the screen: the logical rectangle is clipped to the pane.
    assert target.physical_rect_to_logical(2300, 0, 200, 200) == QtCore.QRect(
        1920, 0, 80, 160
    )
    assert target.physical_rect_to_logical(0, 0, 0, 10) == QtCore.QRect()


def test_capture_region_round_trip():
    region = CaptureRegion(
        physical_left=100,
        physical_top=200,
        physical_width=800,
        physical_height=400,
        logical_left=50,
        logical_top=100,
        logical_width=400,
        logical_height=200,
        scale_x=2.0,
        scale_y=2.0,
        screen_name="DISPLAY1",
    )
    restored = CaptureRegion.from_dict(region.to_dict())
    assert restored == region
    assert region.physical_subrect(QtCore.QRect(150, 150, 200, 100)) == QtCore.QRect(
        200, 100, 400, 200
    )
    assert region.physical_subrect(QtCore.QRect(0, 0, 10, 10)) is None


def test_virtual_desktop_union_with_negative_origin():
    rect = virtual_desktop_rect(
        [
            QtCore.QRect(0, 0, 1920, 1080),
            QtCore.QRect(-1920, 0, 1920, 1080),
        ]
    )
    assert rect == QtCore.QRect(-1920, 0, 3840, 1080)


def test_pair_screens_with_two_physical_monitors():
    screens = [
        SimpleNamespace(
            name=lambda: "LEFT",
            geometry=lambda: QtCore.QRect(0, 0, 1920, 1080),
            devicePixelRatio=lambda: 1.0,
        ),
        SimpleNamespace(
            name=lambda: "RIGHT",
            geometry=lambda: QtCore.QRect(1920, 0, 2560, 1440),
            devicePixelRatio=lambda: 1.0,
        ),
    ]
    monitors = [
        {"left": 1920, "top": 0, "width": 2560, "height": 1440},
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
    ]
    targets = _pair_with_monitors(screens, [dict(item) for item in monitors])
    by_name = {target.name: target for target in targets}
    assert by_name["LEFT"].physical_left == 0
    assert by_name["LEFT"].physical_width == 1920
    assert by_name["RIGHT"].physical_left == 1920
    assert by_name["RIGHT"].physical_width == 2560


def test_pair_screens_with_fractional_dpi():
    screens = [
        SimpleNamespace(
            name=lambda: "SCALED",
            geometry=lambda: QtCore.QRect(0, 0, 1536, 864),
            devicePixelRatio=lambda: 1.25,
        )
    ]
    monitors = [{"left": 0, "top": 0, "width": 1920, "height": 1080}]
    targets = _pair_with_monitors(screens, monitors)
    target = targets[0]
    assert target.physical_width == 1920
    assert target.scale_x == 1.25
from __future__ import annotations

from client.services.target_detector import _containing
from client.services.ui_target import (
    CONTROL,
    ELEMENT,
    SCREEN,
    WINDOW,
    PhysicalRect,
    TargetStack,
    UITarget,
    WindowScan,
    normalize_targets,
    screen_target,
)
from client.services.win32_targets import select_uia_seed


def test_physical_rect_geometry_and_intersection():
    rect = PhysicalRect(10, 20, 100, 50)
    assert rect.right == 110
    assert rect.bottom == 70
    assert rect.area == 5000
    assert rect.contains(10, 20)
    assert rect.contains(109, 69)
    assert not rect.contains(110, 69)
    assert rect.intersection(PhysicalRect(50, 0, 200, 100)) == PhysicalRect(
        50, 20, 60, 50
    )
    assert not rect.intersection(PhysicalRect(0, 0, 5, 5))


def test_normalize_prefers_innermost_and_dedupes_rects():
    screen = PhysicalRect(0, 0, 1920, 1080)
    targets = [
        screen_target(screen, "屏幕 1"),
        UITarget(PhysicalRect(100, 100, 400, 300), "Chrome", WINDOW, "win32", 1),
        UITarget(PhysicalRect(120, 120, 80, 30), "按钮", CONTROL, "win32", 2),
        UITarget(PhysicalRect(120, 120, 80, 30), "确定", ELEMENT, "uia", 0),
    ]
    result = normalize_targets(targets, screen=screen, screen_label="屏幕 1")
    assert [item.rect for item in result] == [
        PhysicalRect(120, 120, 80, 30),
        PhysicalRect(100, 100, 400, 300),
        screen,
    ]
    assert result[0].label == "确定"
    assert result[-1].kind == SCREEN


def test_normalize_clips_to_screen_and_drops_slivers():
    screen = PhysicalRect(0, 0, 800, 600)
    targets = [
        UITarget(PhysicalRect(700, 500, 300, 300), "跨屏窗口", WINDOW, "win32"),
        UITarget(PhysicalRect(100, 100, 1, 50), "分隔条", CONTROL, "win32"),
        UITarget(PhysicalRect(-30, -20, 60, 40), "角落控件", CONTROL, "win32"),
    ]
    result = normalize_targets(targets, screen=screen, min_side=4)
    clipped = next(item for item in result if item.label == "跨屏窗口")
    assert clipped.rect == PhysicalRect(700, 500, 100, 100)
    assert all(item.rect.width >= 4 and item.rect.height >= 4 for item in result)
    # The 1px splitter is gone, but the clipped corner control survives.
    assert any(item.label == "角落控件" for item in result)


def test_fullscreen_window_replaces_duplicate_screen_layer():
    screen = PhysicalRect(0, 0, 1920, 1080)
    result = normalize_targets(
        [UITarget(screen, "全屏游戏", WINDOW, "win32")],
        screen=screen,
    )
    assert len(result) == 1
    assert result[0].kind == WINDOW
    assert result[0].label == "全屏游戏"


def test_target_stack_preserves_selection_by_rect():
    inner = UITarget(PhysicalRect(10, 10, 20, 20), "内", CONTROL)
    outer = UITarget(PhysicalRect(0, 0, 200, 200), "外", WINDOW)
    stack = TargetStack()
    stack.update([inner, outer])
    assert stack.current is inner
    stack.cycle(1)
    assert stack.current is outer
    stack.cycle(1)
    assert stack.current is inner
    # UIA enrichment may add another sibling before the current selection.
    # The user has explicitly wheeled to the outer layer, so the rectangle
    # match keeps it instead of snapping back to the new innermost layer.
    stack.cycle(1)
    assert stack.current is outer
    tiny = UITarget(PhysicalRect(15, 15, 5, 5), "更内", ELEMENT)
    stack.update([tiny, inner, outer], preserve=True)
    assert stack.current is outer
    stack.update([tiny, outer])
    assert stack.current is tiny


def test_target_stack_default_follows_new_innermost_until_user_cycles():
    window = UITarget(PhysicalRect(0, 0, 400, 300), "窗口", WINDOW)
    page = UITarget(PhysicalRect(10, 10, 380, 280), "网页内容", CONTROL)
    stack = TargetStack()
    stack.update([page, window])
    assert stack.current is page
    # The first Win32 pass only knows the page; the UIA pass finds the text
    # control under the pointer, and the default selection must follow it.
    text = UITarget(PhysicalRect(40, 40, 120, 20), "段落", ELEMENT)
    stack.update([text, page, window], preserve=True)
    assert stack.current is text
    # Once the user has wheeled to an outer layer, UIA enrichment keeps it.
    stack.cycle(1)
    assert stack.current is page
    deeper = UITarget(PhysicalRect(45, 42, 60, 16), "文字", ELEMENT)
    stack.update([deeper, text, page, window], preserve=True)
    assert stack.current is page
    # A fresh detection (no preserve) still falls back to the innermost layer.
    stack.update([deeper, text, page, window])
    assert stack.current is deeper


def test_cached_uia_targets_are_filtered_by_pointer_position():
    inside = UITarget(PhysicalRect(10, 10, 20, 20), "内", ELEMENT)
    outside = UITarget(PhysicalRect(100, 100, 20, 20), "旧控件", ELEMENT)
    assert _containing([outside, inside], 15, 15) == [inside]
    assert _containing([outside], 15, 15) == []


def test_win32_select_uia_seed_prefers_web_renderer():
    web = WindowScan(
        top_hwnd=10,
        top_class="Chrome_WidgetWin_1",
        deep_hwnd=11,
        deep_class="Chrome_RenderWidgetHostHWND",
    )
    native = WindowScan(
        top_hwnd=20,
        top_class="OtrProbeWindow",
        deep_hwnd=21,
        deep_class="Button",
    )
    assert select_uia_seed(web) == 11
    assert select_uia_seed(native) == 20
    assert select_uia_seed(WindowScan()) == 0
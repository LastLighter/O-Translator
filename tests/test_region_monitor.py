from __future__ import annotations

import numpy as np
from PySide6 import QtCore

from client.services.region_monitor import (
    ChangeDetector,
    FrameAnalyzer,
    MonitorSettings,
    RegionMonitor,
)
from client.services.screen_capture import CaptureRegion


def _frame(offset: int = 0, size: int = 100) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = min(size - 1, 10 + offset)
    y1 = min(size, 42 + offset)
    frame[y0:y1, 10:42] = 255
    return frame


def _gray(offset: int = 0) -> np.ndarray:
    return FrameAnalyzer.prepare(_frame(offset), max_side=200)


def _settings(**overrides) -> MonitorSettings:
    values = {
        "threshold_percent": 5.0,
        "poll_interval_ms": 100,
        "debounce_ms": 400,
        "cooldown_ms": 2000,
        "noise_threshold": 12,
        "min_changed_pixels": 20,
        "static_percent": 0.8,
        "diff_max_side": 200,
    }
    values.update(overrides)
    return MonitorSettings(**values).normalized()


def test_frame_difference_ignores_subpixel_noise():
    first = np.zeros((100, 100), dtype=np.uint8)
    second = first.copy()
    second[0, :5] = 255
    ratio, changed = FrameAnalyzer.difference(
        first, second, noise_threshold=12, min_changed_pixels=80
    )
    assert ratio == 0.0
    assert changed == 0


def test_frame_difference_counts_real_change():
    first = np.zeros((100, 100), dtype=np.uint8)
    second = first.copy()
    second[10:40, 10:40] = 255
    ratio, changed = FrameAnalyzer.difference(
        first, second, noise_threshold=12, min_changed_pixels=80
    )
    assert changed == 900
    assert ratio == 9.0


def test_change_detector_triggers_after_debounce():
    detector = ChangeDetector(_settings())
    initial = detector.update(_gray(0), 0.0)
    assert initial.initialized
    changed = detector.update(_gray(20), 0.1)
    assert changed.pending
    assert not changed.trigger
    still = detector.update(_gray(20), 0.2)
    assert still.pending
    triggered = detector.update(_gray(20), 0.6)
    assert triggered.trigger


def test_change_detector_waits_for_animation_to_settle():
    detector = ChangeDetector(_settings())
    detector.update(_gray(0), 0.0)
    detector.update(_gray(20), 0.1)
    moving = detector.update(_gray(35), 0.2)
    assert moving.pending
    assert not moving.trigger
    moving2 = detector.update(_gray(60), 0.4)
    assert moving2.pending
    assert not moving2.trigger
    settled = detector.update(_gray(60), 0.9)
    assert settled.trigger


def test_change_detector_respects_cooldown():
    detector = ChangeDetector(_settings(cooldown_ms=2000))
    detector.update(_gray(0), 0.0)
    detector.update(_gray(20), 0.1)
    assert detector.update(_gray(20), 0.6).trigger

    detector.update(_gray(45), 1.0)
    during_cooldown = detector.update(_gray(45), 1.5)
    assert during_cooldown.pending
    assert not during_cooldown.trigger
    after_cooldown = detector.update(_gray(45), 2.7)
    assert after_cooldown.trigger


def test_monitor_settings_normalize_bounds():
    settings = MonitorSettings(
        threshold_percent=0,
        poll_interval_ms=5,
        debounce_ms=5,
        cooldown_ms=5,
        noise_threshold=0,
        min_changed_pixels=0,
        static_percent=99,
        diff_max_side=5,
    ).normalized()
    assert settings.threshold_percent == 0.5
    assert settings.poll_interval_ms == 100
    assert settings.debounce_ms == 100
    assert settings.cooldown_ms == 300
    assert settings.noise_threshold == 2
    assert settings.min_changed_pixels == 1
    assert settings.static_percent == 10.0
    assert settings.diff_max_side == 240


def test_monitor_exclusion_masks_overlay_area():
    monitor = RegionMonitor()
    region = CaptureRegion(
        physical_left=100,
        physical_top=100,
        physical_width=800,
        physical_height=400,
        logical_left=50,
        logical_top=50,
        logical_width=400,
        logical_height=200,
        scale_x=2.0,
        scale_y=2.0,
        screen_name="DISPLAY1",
    )
    gray = np.full((200, 400), 255, dtype=np.uint8)
    overlay_rect = QtCore.QRect(150, 100, 100, 50)
    monitor._apply_exclusion(gray, region, overlay_rect)
    physical = region.physical_subrect(overlay_rect)
    x0 = int(physical.left() * gray.shape[1] / region.physical_width)
    y0 = int(physical.top() * gray.shape[0] / region.physical_height)
    assert gray[y0, x0] == 0
    assert gray[0, 0] == 255


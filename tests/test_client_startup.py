"""Regression tests for tray-first startup and hotkey captures."""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from client.main import MainWindow, _present_initial_window


def test_normal_launch_stays_in_the_tray() -> None:
    events: list[str] = []
    window = SimpleNamespace(
        show=lambda: events.append("show"),
        notify_started=lambda: events.append("tray"),
    )
    _present_initial_window(window, show_window=False)  # type: ignore[arg-type]
    assert events == ["tray"]


def test_show_flag_presents_the_workbench() -> None:
    events: list[str] = []
    window = SimpleNamespace(
        show=lambda: events.append("show"),
        notify_started=lambda: events.append("tray"),
    )
    _present_initial_window(window, show_window=True)  # type: ignore[arg-type]
    assert events == ["show"]


def test_busy_hotkey_capture_keeps_the_workbench_hidden() -> None:
    restored: list[bool] = []
    notices: list[str] = []
    window = SimpleNamespace(
        _busy=True,
        show_status=lambda *args, **kwargs: None,
        _notify_tray=lambda message, **kwargs: notices.append(message),
        _restore_window=lambda: restored.append(True),
    )
    MainWindow._begin_selection(window, "area")  # type: ignore[arg-type]
    assert restored == []
    assert notices == ["当前任务仍在处理中，请稍候。"]

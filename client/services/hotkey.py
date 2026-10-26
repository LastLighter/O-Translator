"""Windows global hotkey support for the desktop client."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import sys
from contextlib import suppress
from typing import Any

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QObject, Signal

from client.config import DEFAULT_HOTKEY


class GlobalHotkey(QObject):
    activated = Signal()

    MOD_ALT = 0x0001
    MOD_CONTROL = 0x0002
    MOD_SHIFT = 0x0004
    MOD_WIN = 0x0008
    MOD_NOREPEAT = 0x4000
    WM_HOTKEY = 0x0312

    def __init__(self, sequence: str = DEFAULT_HOTKEY, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.sequence = (sequence or DEFAULT_HOTKEY).strip().lower()
        self.registered = False
        self.hotkey_id = 0xB001
        self._filter: Any = None
        self._mods, self._vk = self._parse(self.sequence)

    @staticmethod
    def _parse(sequence: str) -> tuple[int, int]:
        mods = 0
        key = "t"
        for part in [item.strip() for item in sequence.replace("+", " ").split()]:
            if part in {"ctrl", "control"}:
                mods |= GlobalHotkey.MOD_CONTROL
            elif part in {"alt", "option"}:
                mods |= GlobalHotkey.MOD_ALT
            elif part == "shift":
                mods |= GlobalHotkey.MOD_SHIFT
            elif part in {"win", "windows", "meta"}:
                mods |= GlobalHotkey.MOD_WIN
            elif part:
                key = part
        if len(key) == 1 and key.isalnum():
            vk = ord(key.upper())
        elif key.startswith("f") and key[1:].isdigit():
            vk = 0x6F + int(key[1:])
        else:
            vk = ord("T")
        return mods, vk

    def register(self) -> bool:
        if sys.platform != "win32":
            return False
        user32 = ctypes.windll.user32
        if not user32.RegisterHotKey(
            None, self.hotkey_id, self._mods | self.MOD_NOREPEAT, self._vk
        ):
            return False
        self.registered = True

        class Filter(QtCore.QAbstractNativeEventFilter):
            def __init__(inner_self, owner: GlobalHotkey) -> None:
                super().__init__()
                inner_self.owner = owner

            def nativeEventFilter(inner_self, event_type, message):
                try:
                    msg = ctypes.wintypes.MSG.from_address(int(message))
                    if (
                        msg.message == GlobalHotkey.WM_HOTKEY
                        and msg.wParam == inner_self.owner.hotkey_id
                    ):
                        inner_self.owner.activated.emit()
                except (ValueError, TypeError, OSError):
                    pass
                return False, 0

        self._filter = Filter(self)
        QtWidgets.QApplication.instance().installNativeEventFilter(self._filter)
        return True

    def unregister(self) -> None:
        if not self.registered:
            return
        with suppress(OSError):
            ctypes.windll.user32.UnregisterHotKey(None, self.hotkey_id)
        if self._filter is not None:
            with suppress(RuntimeError, ValueError):
                QtWidgets.QApplication.instance().removeNativeEventFilter(self._filter)
        self._filter = None
        self.registered = False

    @property
    def display(self) -> str:
        return " + ".join(part.title() for part in self.sequence.split("+"))

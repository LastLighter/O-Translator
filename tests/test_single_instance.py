from __future__ import annotations

import os
import uuid

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets
from PySide6.QtTest import QTest

from client.services.single_instance import SingleInstanceGuard


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def _guard(tmp_path, scope: str) -> SingleInstanceGuard:
    return SingleInstanceGuard(
        app_id="OTRTest",
        scope=scope,
        state_dir=tmp_path,
    )


def test_second_guard_cannot_acquire_and_release_reopens(qapp, tmp_path):
    scope = uuid.uuid4().hex
    first = _guard(tmp_path, scope)
    second = _guard(tmp_path, scope)
    third = _guard(tmp_path, scope)
    try:
        assert first.acquire() is True
        assert second.acquire() is False
        first.release()
        assert third.acquire() is True
    finally:
        second.release()
        third.release()
        first.release()


def test_extra_launch_notifies_the_first_instance(qapp, tmp_path):
    scope = uuid.uuid4().hex
    first = _guard(tmp_path, scope)
    second = _guard(tmp_path, scope)
    activated: list[bool] = []
    first.activation_requested.connect(lambda: activated.append(True))
    try:
        assert first.acquire() is True
        assert second.acquire() is False
        assert second.notify_existing(1500) is True
        QTest.qWait(250)
        assert activated == [True]
    finally:
        second.release()
        first.release()

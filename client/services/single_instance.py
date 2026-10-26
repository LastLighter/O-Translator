"""Single-instance guard for the desktop client.

Clicking the executable repeatedly must never start a second tray icon, overlay
stack, global hotkey registration, or backend process.  Qt's ``QLockFile`` gives
us an atomic, per-user lock; ``QLocalServer``/``QLocalSocket`` provide a small
IPC channel so the already-running instance can bring its main window forward
instead of silently ignoring the click.

The guard is deliberately scoped to the application root: two different builds
(e.g. a development checkout and a packaged release) may run side by side, while
every launch of the same build maps to the same lock and pipe name.
"""

from __future__ import annotations

import getpass
import hashlib
import os
import re
from contextlib import suppress
from pathlib import Path

from PySide6 import QtCore, QtNetwork

from client.config import APP_NAME, PROJECT_ROOT


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "O-Translator"


def _app_scope(scope: str | None = None) -> str:
    if scope:
        return _safe_component(scope)
    try:
        user = getpass.getuser()
    except Exception:  # pragma: no cover - unusual Windows service contexts
        user = "user"
    root = os.path.normcase(str(PROJECT_ROOT))
    digest = hashlib.sha1(f"{root}|{user}".encode("utf-8", "replace")).hexdigest()
    return digest[:16]


class SingleInstanceGuard(QtCore.QObject):
    """Own the process lock and notify the first instance on extra launches."""

    activation_requested = QtCore.Signal()

    def __init__(
        self,
        *,
        app_id: str = APP_NAME,
        scope: str | None = None,
        state_dir: str | Path | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._app_id = _safe_component(app_id)
        self._scope = _app_scope(scope)
        self._state_dir_arg = Path(state_dir) if state_dir is not None else None
        self._lock: QtCore.QLockFile | None = None
        self._server: QtNetwork.QLocalServer | None = None
        self._acquired = False

    # -------------------------------------------------------------- properties
    @property
    def acquired(self) -> bool:
        return self._acquired

    @property
    def scope(self) -> str:
        return self._scope

    @property
    def server_name(self) -> str:
        return f"{self._app_id}-Desktop-{self._scope}"

    @property
    def lock_path(self) -> Path:
        return self._state_dir() / f"single-instance-{self._scope}.lock"

    def _state_dir(self) -> Path:
        if self._state_dir_arg is not None:
            directory = self._state_dir_arg
        else:
            base = QtCore.QStandardPaths.writableLocation(
                QtCore.QStandardPaths.StandardLocation.AppLocalDataLocation
            )
            if not base:
                base = QtCore.QStandardPaths.writableLocation(
                    QtCore.QStandardPaths.StandardLocation.TempLocation
                )
            directory = Path(base) / "instance"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    # ------------------------------------------------------------------ acquire
    def acquire(self) -> bool:
        """Try to become the one running instance.

        Returns ``True`` when this process owns the lock.  ``False`` means a
        live instance already exists; callers should notify it and exit.
        """
        if self._acquired:
            return True
        try:
            lock_path = self.lock_path
        except OSError:  # pragma: no cover - unwritable profile
            # Never prevent the app from starting just because the lock path is
            # unavailable; uniqueness is best-effort in that situation.
            self._acquired = True
            return True
        lock = QtCore.QLockFile(str(lock_path))
        # QLockFile checks the owning PID, so a long-running instance is never
        # mistaken for a stale lock.  A short stale window still recovers from a
        # crash without asking the user to delete files.
        lock.setStaleLockTime(10_000)
        if not lock.tryLock(0):
            return False
        self._lock = lock
        self._acquired = True
        self._listen()
        return True

    def _listen(self) -> None:
        # A previously crashed process can leave a pipe behind on some Qt
        # versions.  Only called after we own the lock, so no live instance is
        # using this name.
        with suppress(OSError, RuntimeError):
            QtNetwork.QLocalServer.removeServer(self.server_name)
        server = QtNetwork.QLocalServer(self)
        server.setSocketOptions(
            QtNetwork.QLocalServer.SocketOption.UserAccessOption
        )
        server.newConnection.connect(self._accept_connections)
        if server.listen(self.server_name):
            self._server = server
            return
        # The pipe exists but we still own the lock (for example an old build
        # without the lock).  Keep running; duplicate launches are still
        # stopped by the lock itself.
        self._server = None

    def notify_existing(self, timeout_ms: int = 1200) -> bool:
        """Ask the running instance to show itself.

        The caller has already failed to acquire the lock.  Retry briefly so a
        second click during the first instance's startup still works once its
        server starts listening.
        """
        deadline = QtCore.QElapsedTimer()
        deadline.start()
        while deadline.elapsed() < max(0, int(timeout_ms)):
            socket = QtNetwork.QLocalSocket()
            # ReadWrite (the default) matters on Windows: a WriteOnly socket
            # can report a successful connect yet fail before the server's
            # named-pipe instance accepts it.
            socket.connectToServer(self.server_name)
            if socket.waitForConnected(200):
                # The connection itself is the activation request; see
                # _accept_connections for why no payload is required.
                socket.disconnectFromServer()
                return True
            remaining = max(0, int(timeout_ms) - deadline.elapsed())
            if remaining <= 0:
                break
            QtCore.QThread.msleep(min(60, remaining))
        return False

    # --------------------------------------------------------------------- IPC
    def _accept_connections(self) -> None:
        server = self._server
        if server is None:
            return
        while server.hasPendingConnections():
            socket = server.nextPendingConnection()
            if socket is None:
                return
            # The connection itself is the activation request.  Relying on a
            # payload here is racy on Windows: the second process may exit
            # before the first one accepts the named-pipe instance, which can
            # drop buffered bytes even though newConnection fired.
            with suppress(OSError, RuntimeError):
                socket.readAll()
                socket.deleteLater()
            self.activation_requested.emit()

    # ----------------------------------------------------------------- release
    def release(self) -> None:
        if self._server is not None:
            with suppress(OSError, RuntimeError):
                self._server.close()
            self._server = None
        if self._acquired:
            with suppress(OSError, RuntimeError):
                QtNetwork.QLocalServer.removeServer(self.server_name)
        if self._lock is not None:
            with suppress(OSError, RuntimeError):
                self._lock.unlock()
            self._lock = None
        self._acquired = False


__all__ = ["SingleInstanceGuard"]

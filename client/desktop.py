"""Backward-compatible entry point.

The desktop application now lives in :mod:`client.main`, split into
``client.ui`` and ``client.services`` modules. Existing commands such as
``python -m client.desktop`` continue to work.
"""

from __future__ import annotations

from client.main import MainWindow, main

__all__ = ["MainWindow", "main"]


if __name__ == "__main__":
    raise SystemExit(main())

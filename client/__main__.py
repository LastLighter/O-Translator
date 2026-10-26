"""O-Translator desktop application entry point."""

from client.main import MainWindow, main

__all__ = ["MainWindow", "main"]

if __name__ == "__main__":
    raise SystemExit(main())

"""Package the PyInstaller output as a portable Windows zip.

The .NET ``ZipFile`` helper on Windows PowerShell writes ``\\`` separators into
entry names, which some extractors do not treat as directories. This script
walks the tree explicitly and writes forward-slash names (and keeps empty
``data``/``models`` folders).

Usage: python scripts/package_zip.py [source_dir] [target.zip] [--exclude DIR]...

``--exclude`` skips a path prefix relative to the source (for example
``--exclude data`` when the source tree has been launched once and contains
logs). The archive root is ``O-Translator/`` by default; override it with
``--root-name`` when packaging a differently named build.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT_NAME = "O-Translator"


def _excluded(relative: Path, excludes: list[str]) -> bool:
    name = relative.as_posix()
    return any(name == item or name.startswith(item + "/") for item in excludes)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default="dist/O-Translator")
    parser.add_argument("target", nargs="?", default="dist/O-Translator-windows-x64.zip")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="DIR",
        help="relative path prefix to skip (repeatable)",
    )
    parser.add_argument(
        "--root-name",
        default=ROOT_NAME,
        help=f"archive root folder name (default: {ROOT_NAME})",
    )
    args = parser.parse_args(argv[1:])

    source = Path(args.source)
    target = Path(args.target)
    root_name = args.root_name.strip("/")
    excludes = [item.strip("/").replace("\\", "/") for item in args.exclude]
    if not source.is_dir():
        raise SystemExit(f"Missing build directory: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    # Level 4 is a good trade-off for a ~1 GB portable tree: the 200 MB ONNX
    # graph is barely compressible, and level 6 nearly doubles archive time.
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if _excluded(relative, excludes):
                continue
            arcname = (Path(root_name) / relative).as_posix()
            if path.is_dir():
                if not any(path.iterdir()):
                    archive.writestr(arcname + "/", "")
            else:
                archive.write(path, arcname)

    size_mb = target.stat().st_size / (1024 * 1024)
    print(f"Wrote {target} ({size_mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

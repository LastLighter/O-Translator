#!/usr/bin/env python3
"""Verify that a packaged O-Translator build is self-contained.

The check works on an extracted onedir build or directly on its zip, without
extracting anything. It answers one question: can a user unzip this folder on a
clean Windows x64 machine and run it, including GPU inference, without fetching
models or CUDA DLLs separately?

Usage:
    python scripts/verify_dist.py dist/O-Translator-autopick
    python scripts/verify_dist.py dist/O-Translator-autopick-windows-x64.zip
    python scripts/verify_dist.py dist/O-Translator --cpu-only
    python scripts/verify_dist.py dist/*.zip

Exit code is 0 only when every artifact passes; warnings do not fail the run.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

MIN_EXE_BYTES = 1 * 1024 * 1024
# backend/config.py treats anything smaller as an HTML error page, not a model.
MIN_LAMA_BYTES = 20 * 1024 * 1024

LAMA_PATTERNS = (
    "_internal/models/lama/*.onnx",
    "models/lama/*.onnx",
)
RAPIDOCR_DIRS = (
    "_internal/models/rapidocr/",
    "_internal/rapidocr/models/",
)
USAGE_NAMES = ("使用说明.txt",)


@dataclass
class Check:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str


@dataclass
class Report:
    artifact: Path
    kind: str
    checks: list[Check] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(check.status == "FAIL" for check in self.checks)

    @property
    def warnings(self) -> int:
        return sum(1 for check in self.checks if check.status == "WARN")


class Artifact:
    """A directory tree or zip archive exposed as relative-path -> size."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.is_zip = path.suffix.lower() == ".zip"
        self.files: dict[str, int] = {}
        self.dirs: set[str] = set()
        self.root_prefix = ""

        if self.is_zip:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
            names = [info.filename for info in infos if info.filename]
            self.root_prefix = _single_root_prefix(names)
            for info in infos:
                name = _strip_prefix(info.filename, self.root_prefix)
                if not name:
                    continue
                if info.is_dir():
                    self.dirs.add(name)
                else:
                    self.files[name] = info.file_size
        else:
            for entry in path.rglob("*"):
                relative = entry.relative_to(path).as_posix()
                if entry.is_dir():
                    self.dirs.add(relative)
                elif entry.is_file():
                    self.files[relative] = entry.stat().st_size

    def glob(self, pattern: str) -> list[tuple[str, int]]:
        return [
            (name, size)
            for name, size in self.files.items()
            if fnmatch.fnmatch(name, pattern)
        ]

    def under(self, prefix: str) -> list[tuple[str, int]]:
        return [
            (name, size)
            for name, size in self.files.items()
            if name.startswith(prefix)
        ]

    def total(self, entries: list[tuple[str, int]]) -> int:
        return sum(size for _, size in entries)


def _single_root_prefix(names: list[str]) -> str:
    """Return ``O-Translator/`` when every zip entry shares that root."""
    roots = {name.split("/", 1)[0] for name in names if name}
    if len(roots) != 1:
        return ""
    root = next(iter(roots))
    if not any(name.startswith(root + "/") for name in names):
        return ""
    return root + "/"


def _strip_prefix(name: str, prefix: str) -> str:
    if prefix and name.startswith(prefix):
        name = name[len(prefix) :]
    return name.rstrip("/")


def _mb(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB"


def _gb(size: int) -> str:
    return f"{size / (1024 * 1024 * 1024):.2f} GB"


def _classify(artifact: Artifact) -> str:
    if "O-Translator.exe" in artifact.files:
        return "bundle"
    providers = [
        name
        for name in artifact.files
        if name.endswith("onnxruntime/capi/onnxruntime_providers_cuda.dll")
    ]
    if providers:
        return "gpu-addon"
    return "unknown"


def _check_executable(artifact: Artifact) -> Check:
    size = artifact.files.get("O-Translator.exe", 0)
    if size >= MIN_EXE_BYTES:
        return Check("executable", "PASS", f"O-Translator.exe ({_mb(size)})")
    if size:
        return Check(
            "executable",
            "FAIL",
            f"O-Translator.exe is only {_mb(size)}; expected > 1 MB",
        )
    return Check("executable", "FAIL", "O-Translator.exe is missing")


def _check_internal(artifact: Artifact) -> Check:
    entries = artifact.under("_internal/")
    if not entries:
        return Check("runtime", "FAIL", "_internal/ is missing or empty")
    return Check(
        "runtime",
        "PASS",
        f"_internal/ ({len(entries)} files, {_gb(artifact.total(entries))})",
    )


def _check_lama(artifact: Artifact) -> Check:
    candidates: list[tuple[str, int]] = []
    for pattern in LAMA_PATTERNS:
        candidates.extend(artifact.glob(pattern))
    if not candidates:
        return Check(
            "LaMa model",
            "FAIL",
            "no ONNX graph under _internal/models/lama/ or models/lama/",
        )
    name, size = max(candidates, key=lambda item: item[1])
    if size < MIN_LAMA_BYTES:
        return Check(
            "LaMa model",
            "FAIL",
            f"{name} is only {_mb(size)}; expected a ~200 MB graph",
        )
    return Check("LaMa model", "PASS", f"{name} ({_mb(size)})")


def _check_ocr(artifact: Artifact) -> Check:
    rapidocr = [
        (name, size)
        for directory in RAPIDOCR_DIRS
        for name, size in artifact.under(directory)
        if name.endswith(".onnx")
    ]
    if rapidocr:
        det = [name for name, _ in rapidocr if "det" in Path(name).name.lower()]
        rec = [name for name, _ in rapidocr if "rec" in Path(name).name.lower()]
        total = artifact.total(rapidocr)
        if not det or not rec:
            return Check(
                "OCR models",
                "FAIL",
                f"RapidOCR graphs found ({len(rapidocr)} files) but no det+rec pair",
            )
        return Check(
            "OCR models",
            "PASS",
            f"RapidOCR det+rec ({len(rapidocr)} files, {_mb(total)})",
        )

    paddle = artifact.under("_internal/paddle/")
    if paddle:
        return Check(
            "OCR models",
            "WARN",
            f"legacy PaddleOCR only ({_gb(artifact.total(paddle))}); "
            "current builds use RapidOCR ONNX",
        )
    return Check(
        "OCR models",
        "FAIL",
        "no RapidOCR graphs under _internal/models/rapidocr/ or "
        "_internal/rapidocr/models/",
    )


def _cuda_entries(artifact: Artifact) -> list[tuple[str, int]]:
    """CUDA DLLs in a full build or in an add-on whose root was stripped."""
    entries: list[tuple[str, int]] = []
    for prefix in ("gpu_runtime/onnxruntime/capi/", "onnxruntime/capi/"):
        entries.extend(artifact.under(prefix))
    return [(name, size) for name, size in entries if name.endswith(".dll")]


def _check_gpu(artifact: Artifact, *, cpu_only: bool) -> Check:
    entries = _cuda_entries(artifact)
    provider = [
        name
        for name, _ in entries
        if Path(name).name.lower() == "onnxruntime_providers_cuda.dll"
    ]
    cudnn = [
        name
        for name, _ in entries
        if re.match(r"cudnn.*_9\.dll$", Path(name).name, re.IGNORECASE)
    ]
    cublas = [
        name
        for name, _ in entries
        if re.match(r"(cublas|cudart).*_12\.dll$", Path(name).name, re.IGNORECASE)
    ]
    total = artifact.total(entries)

    if provider and cudnn and cublas:
        return Check(
            "CUDA runtime",
            "PASS",
            "gpu_runtime/onnxruntime/capi "
            f"(provider + {len(cudnn)} cuDNN + {len(cublas)} cuBLAS/cudart DLLs, "
            f"{_gb(total)})",
        )

    missing: list[str] = []
    if not provider:
        missing.append("onnxruntime_providers_cuda.dll")
    if not cudnn:
        missing.append("cudnn*_9.dll")
    if not cublas:
        missing.append("cublas*_12.dll/cudart*_12.dll")
    detail = "missing " + ", ".join(missing)
    if entries:
        detail += f"; found {len(entries)} DLLs in gpu_runtime/"
    else:
        detail += "; gpu_runtime/ is absent"
    return Check("CUDA runtime", "WARN" if cpu_only else "FAIL", detail)


def _check_usage(artifact: Artifact) -> Check:
    for name in USAGE_NAMES:
        if name in artifact.files:
            return Check("usage notes", "PASS", name)
    return Check("usage notes", "WARN", "使用说明.txt is missing")


def verify(path: Path, *, cpu_only: bool) -> Report:
    artifact = Artifact(path)
    kind = _classify(artifact)
    report = Report(path, kind)

    if kind == "gpu-addon":
        report.checks.append(
            Check(
                "add-on layout",
                "PASS",
                "gpu_runtime/onnxruntime/capi detected",
            )
        )
        gpu = _check_gpu(artifact, cpu_only=False)
        report.checks.append(gpu)
        return report

    if kind == "unknown":
        report.checks.append(
            Check(
                "layout",
                "FAIL",
                "no O-Translator.exe and no gpu_runtime/ provider DLL; "
                "not a build or GPU add-on",
            )
        )
        return report

    report.checks.append(_check_executable(artifact))
    report.checks.append(_check_internal(artifact))
    report.checks.append(_check_lama(artifact))
    report.checks.append(_check_ocr(artifact))
    report.checks.append(_check_gpu(artifact, cpu_only=cpu_only))
    report.checks.append(_check_usage(artifact))
    return report


def _print_report(report: Report) -> None:
    suffix = ", zip" if report.artifact.suffix.lower() == ".zip" else ""
    print(f"\n=== {report.artifact} [{report.kind}{suffix}] ===")
    width = max((len(check.name) for check in report.checks), default=0)
    for check in report.checks:
        print(f"  [{check.status:4}] {check.name:<{width}}  {check.detail}")
    if report.failed:
        print("  RESULT: FAIL")
    elif report.warnings:
        print(f"  RESULT: PASS with {report.warnings} warning(s)")
    else:
        print("  RESULT: PASS")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Check that a packaged O-Translator build is self-contained.",
    )
    parser.add_argument(
        "artifacts",
        nargs="+",
        type=Path,
        help="Extracted build folder or portable zip",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Explicitly allow a package without the gpu_runtime/ CUDA add-on",
    )
    args = parser.parse_args(argv[1:])

    reports: list[Report] = []
    for path in args.artifacts:
        if not path.exists():
            print(f"Missing artifact: {path}", file=sys.stderr)
            return 2
        reports.append(verify(path, cpu_only=args.cpu_only))

    for report in reports:
        _print_report(report)

    failed = [report for report in reports if report.failed]
    if failed:
        print(f"\n{len(failed)} artifact(s) failed the completeness check.")
        return 1
    print(f"\nAll {len(reports)} artifact(s) passed the completeness check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

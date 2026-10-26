"""Download optional model weights for O-Translator.

Usage:
    python scripts/download_models.py --lama
    python scripts/download_models.py --lama --proxy http://127.0.0.1:7897
    python scripts/download_models.py --lama --url https://your-mirror/lama_fp32.onnx
    python scripts/download_models.py --check

The default source is the Apache-2.0 Carve/LaMa-ONNX export. A Hugging Face
mirror is tried first because it is often reachable where huggingface.co is not.
Set ``OTR_PROXY``/``HTTPS_PROXY`` or ``OTR_HF_ENDPOINT`` to reuse the server's
network configuration. No weights are bundled with the repository.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = ROOT / "models" / "lama" / "lama_fp32.onnx"
OPTIMIZED_NAMES = ("lama_fp32_sim.onnx", "lama_fp32_fp16.onnx", "lama_fp32_int8.onnx")
CANDIDATES = [
    "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
]
MIN_EXPECTED_BYTES = 20 * 1024 * 1024


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _download(url: str, target: Path, proxy: str | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "O-Translator/0.1 (+https://github.com/)",
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
        },
    )
    proxies = (
        {"http": proxy, "https": proxy}
        if proxy
        else dict(urllib.request.getproxies())
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(proxies)
    )
    print(f"Downloading {url}")
    with opener.open(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        suffix = ".part"
        fd, temporary = tempfile.mkstemp(
            prefix="lama-", suffix=suffix, dir=str(target.parent)
        )
        downloaded = 0
        last_print = 0.0
        try:
            with os.fdopen(fd, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_print > 0.4:
                        last_print = now
                        if total:
                            percent = downloaded / total * 100
                            print(
                                f"\r  {percent:5.1f}%  {_human(downloaded)} / {_human(total)}",
                                end="",
                                flush=True,
                            )
                        else:
                            print(f"\r  {_human(downloaded)}", end="", flush=True)
            print()
            if downloaded < MIN_EXPECTED_BYTES:
                raise RuntimeError(
                    f"Downloaded file is only {_human(downloaded)}; expected a LaMa ONNX "
                    "model of roughly 200 MB. The mirror may have returned an HTML error page."
                )
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def check_model(path: Path) -> bool:
    if not path.is_file():
        print(f"Missing: {path}")
        return False
    print(f"Model: {path} ({_human(path.stat().st_size)})")
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is not installed; cannot inspect the graph.")
        return False
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        print(f"ONNX Runtime {ort.__version__}")
        for item in inputs:
            print(f"  input  {item.name}: {item.shape} {item.type}")
        for item in session.get_outputs():
            print(f"  output {item.name}: {item.shape} {item.type}")
        if len(inputs) < 2:
            print("WARNING: LaMa expects separate image and mask inputs.")
            return False
        return True
    except Exception as exc:
        print(f"Failed to load model: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lama", action="store_true", help="download LaMa ONNX weights")
    parser.add_argument("--check", action="store_true", help="inspect an existing model")
    parser.add_argument("--url", help="override the LaMa download URL")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="fold the graph with onnxsim after downloading (recommended)",
    )
    parser.add_argument(
        "--proxy",
        default=os.environ.get("OTR_PROXY") or os.environ.get("HTTPS_PROXY"),
        help="proxy URL, e.g. http://127.0.0.1:7897",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()

    if not args.lama and not args.check:
        parser.print_help()
        return 0

    target = args.output
    if args.check:
        for name in OPTIMIZED_NAMES:
            optimized = target.with_name(name)
            if optimized.is_file():
                print(f"Optimized model present: {optimized}")
                return 0 if check_model(optimized) else 1
        return 0 if check_model(target) else 1

    for name in OPTIMIZED_NAMES:
        optimized = target.with_name(name)
        if optimized.is_file() and optimized.stat().st_size >= MIN_EXPECTED_BYTES:
            print(f"Optimized model already present: {optimized}")
            print("No download needed; the inpainter prefers this graph.")
            return 0

    if target.is_file() and target.stat().st_size >= MIN_EXPECTED_BYTES:
        print(f"Already present: {target} ({_human(target.stat().st_size)})")
        if args.optimize:
            import subprocess

            return subprocess.call(
                [sys.executable, str(ROOT / "scripts" / "optimize_lama.py"), "--verify"]
            )
        return 0 if check_model(target) else 1

    urls = [args.url] if args.url else CANDIDATES
    endpoint = os.environ.get("OTR_HF_ENDPOINT") or os.environ.get("HF_ENDPOINT")
    if endpoint and not args.url:
        mirror = f"{endpoint.rstrip('/')}/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
        urls = [mirror] + [url for url in urls if url != mirror]
    last_error: Exception | None = None
    for url in urls:
        try:
            _download(url, target, args.proxy)
            if args.optimize:
                import subprocess

                return subprocess.call(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "optimize_lama.py"),
                        "--verify",
                    ]
                )
            return 0 if check_model(target) else 1
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            print(f"\nFailed: {exc}", file=sys.stderr)
            if url != urls[-1]:
                print("Trying the next source…", file=sys.stderr)
    print(
        f"\nCould not download LaMa. Last error: {last_error}\n"
        "Download lama_fp32.onnx manually and save it to "
        f"{target}, or use --url with a reachable mirror.\n"
        "O-Translator still runs with the OpenCV Telea fallback.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
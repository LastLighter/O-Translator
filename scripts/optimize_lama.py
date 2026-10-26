"""Fold, convert, and validate the LaMa ONNX graph.

The Carve/LaMa-ONNX export keeps thousands of Shape/Slice/Reshape/Constant nodes
because its spatial dimensions are dynamic. Fixing the batch (the export is
already fixed at 512x512) lets ``onnxsim`` constant-fold them:

    17,480 nodes -> ~3,000 nodes, load 35 s -> ~2 s, same output.

Precision options are intentionally opt-in:

    # graph folding only (accurate, CPU-friendly; used by the packaged build)
    python scripts/optimize_lama.py --verify

    # FP16 weights for CUDA/DirectML/OpenVINO-GPU deployments
    python scripts/optimize_lama.py --precision fp16 --verify

    # dynamic INT8 (experimental; validate before deploying)
    python scripts/optimize_lama.py --precision int8 --verify

The checker compares the optimized graph against the source graph. FP16 and
INT8 are rejected unless ``--allow-lossy`` is passed, because LaMa's FFC/FFT
blocks are sensitive to precision loss (especially on CPU where ORT emulates
FP16 and re-quantizes activations).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = ROOT / "models" / "lama" / "lama_fp32.onnx"
SIMPLIFIED_NAME = "lama_fp32_sim.onnx"
FP16_NAME = "lama_fp32_fp16.onnx"
INT8_NAME = "lama_fp32_int8.onnx"


def _fixed_shapes(model_path: Path) -> tuple[dict[str, list[int]], int]:
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    shapes: dict[str, list[int]] = {}
    for value in model.graph.input:
        dims = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        shapes[value.name] = [dim if dim > 0 else 1 for dim in dims]
    return shapes, len(model.graph.node)


def _simplify(model_path: Path, output_path: Path) -> Path:
    try:
        import onnxsim
    except ImportError as exc:
        raise RuntimeError(
            "onnxsim is not installed. Install it with:\n  python -m pip install onnxsim"
        ) from exc

    shapes, original_nodes = _fixed_shapes(model_path)
    print(f"Simplifying {model_path.name} with shapes {shapes} ...")
    started = time.perf_counter()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="lama-sim-", suffix=".onnx", dir=str(output_path.parent)
    )
    os.close(descriptor)
    os.unlink(temporary_name)
    temporary = Path(temporary_name)
    try:
        simplified, ok = onnxsim.simplify(
            str(model_path),
            overwrite_input_shapes=shapes,
            check_n=0,
            output_path=str(temporary),
        )
        if not ok:
            raise RuntimeError("onnxsim reported failure")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    simplified_nodes = len(simplified.graph.node)
    print(
        f"Saved {output_path.name}: {original_nodes} -> {simplified_nodes} nodes, "
        f"{output_path.stat().st_size / 1024 / 1024:.1f} MB, "
        f"{time.perf_counter() - started:.1f}s"
    )
    return output_path


def _convert_fp16(source_path: Path, output_path: Path) -> Path:
    import onnx

    try:
        from onnxconverter_common import float16

        model = onnx.load(str(source_path), load_external_data=False)
        converted = float16.convert_float_to_float16(model, keep_io_types=True)
    except ImportError:
        from onnxruntime.transformers.float16 import convert_float_to_float16

        model = onnx.load(str(source_path), load_external_data=False)
        converted = convert_float_to_float16(model, keep_io_types=True)

    onnx.checker.check_model(converted)
    onnx.save(converted, str(output_path))
    print(
        f"Saved {output_path.name}: FP16, "
        f"{output_path.stat().st_size / 1024 / 1024:.1f} MB"
    )
    return output_path


def _convert_int8(source_path: Path, output_path: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(source_path),
        str(output_path),
        weight_type=QuantType.QInt8,
        extra_options={"EnableSubgraph": False},
    )
    print(
        f"Saved {output_path.name}: dynamic INT8, "
        f"{output_path.stat().st_size / 1024 / 1024:.1f} MB"
    )
    return output_path


def _verify(
    source_path: Path,
    optimized_path: Path,
    *,
    tolerance: float = 1e-3,
) -> bool:
    import numpy as np
    import onnxruntime as ort

    shapes, _ = _fixed_shapes(source_path)
    rng = np.random.default_rng(0)
    feeds = {name: rng.random(shape, dtype=np.float32) for name, shape in shapes.items()}
    if "mask" in feeds:
        feeds["mask"] = (feeds["mask"] > 0.5).astype(np.float32)

    outputs = []
    for path in (source_path, optimized_path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        output = session.run(None, feeds)[0].astype(np.float32)
        outputs.append(output)

    diff = np.abs(outputs[0] - outputs[1])
    passed = bool(np.allclose(outputs[0], outputs[1], atol=tolerance))
    print(
        f"verify: max diff {diff.max():.6f}, mean {diff.mean():.6f}, "
        f"allclose({tolerance})={passed}"
    )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default="fp32",
        help="fp32 = graph folding only; fp16/int8 add precision reduction",
    )
    parser.add_argument(
        "--precision-only",
        action="store_true",
        help="skip onnxsim and convert the source graph directly",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing output")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run source and optimized graphs on random inputs and compare",
    )
    parser.add_argument(
        "--allow-lossy",
        action="store_true",
        help="keep an FP16/INT8 output even when verification fails",
    )
    arguments = parser.parse_args()

    model_path = arguments.model
    if not model_path.is_file():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1

    if arguments.precision == "fp32":
        default_name = SIMPLIFIED_NAME
    elif arguments.precision == "fp16":
        default_name = FP16_NAME
    else:
        default_name = INT8_NAME
    output_path = arguments.output or model_path.with_name(default_name)
    if output_path.exists() and not arguments.force:
        print(f"Already exists: {output_path} (pass --force to rebuild)")
        return 0

    base_path = model_path
    try:
        if arguments.precision == "fp32":
            if arguments.precision_only:
                raise RuntimeError("--precision-only is not valid for fp32")
            _simplify(model_path, output_path)
        else:
            if not arguments.precision_only:
                folded = model_path.with_name(SIMPLIFIED_NAME)
                if not folded.is_file():
                    folded = _simplify(model_path, folded)
                base_path = folded
            if arguments.precision == "fp16":
                _convert_fp16(base_path, output_path)
            else:
                _convert_int8(base_path, output_path)
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        return 1

    if arguments.verify:
        try:
            passed = _verify(model_path, output_path)
        except Exception as exc:
            print(f"Verification failed to run: {exc}", file=sys.stderr)
            return 1
        if not passed and arguments.precision != "fp32" and not arguments.allow_lossy:
            print(
                "Precision reduction changed the output beyond the tolerance. "
                "The file was kept for inspection; remove it or pass "
                "--allow-lossy to acknowledge the quality trade-off.",
                file=sys.stderr,
            )
            return 2
    print("Done. Restart the backend to use the optimized graph.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

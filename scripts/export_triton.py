#!/usr/bin/env python3
"""
Export the trained PyTorch model to ONNX and place it in the Triton model repository.

Run this once (offline) instead of rebuilding the engine on every container start:

    python scripts/export_triton.py

Output: model_repository/classifier/1/model.onnx

Requires `optimum[onnxruntime]` + `torch` (not needed by the inference gateway at runtime).
"""

import sys
import shutil
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from loguru import logger

from src.config import settings, PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to ONNX for Triton")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to the trained model (default: settings.models_dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "model_repository" / "classifier" / "1",
        help="Triton model version directory to write model.onnx into",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Convert the ONNX graph to FP16 (halves GPU memory, faster). IO stays FP32.",
    )
    return parser.parse_args()


def _to_fp16(onnx_path: Path) -> None:
    """Fuse transformer kernels and convert to FP16 in place (FP32 graph IO kept).

    Uses ONNX Runtime's transformer optimizer (model_type="bert" works for this
    CamemBERT/UmBERTo model) rather than onnxconverter_common, which produces an
    invalid Cast-typed graph here that Triton refuses to load.
    """
    from onnxruntime.transformers import optimizer
    from onnxruntime.transformers.fusion_options import FusionOptions

    logger.info("Fusing transformer kernels (model_type=bert) and converting to FP16...")
    opt = optimizer.optimize_model(
        str(onnx_path),
        model_type="bert",
        num_heads=12,
        hidden_size=768,
        optimization_options=FusionOptions("bert"),
    )
    opt.convert_float_to_float16(keep_io_types=True)
    opt.save_model_to_file(str(onnx_path))
    logger.info("FP16 conversion done")


def main():
    args = parse_args()

    model_path = args.model_path or settings.models_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.onnx"

    logger.info(f"Loading and exporting model to ONNX from {model_path}")

    # Lazy import so the gateway runtime doesn't need optimum/torch.
    from optimum.onnxruntime import ORTModelForSequenceClassification

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        model = ORTModelForSequenceClassification.from_pretrained(
            model_path,
            export=True,
        )
        model.save_pretrained(tmp_dir)

        exported = tmp_dir / "model.onnx"
        if not exported.exists():
            # Some optimum versions name it differently; pick the only .onnx file.
            candidates = list(tmp_dir.glob("*.onnx"))
            if not candidates:
                raise FileNotFoundError(f"No .onnx produced in {tmp_dir}")
            exported = candidates[0]

        shutil.copy2(exported, output_path)

    if args.fp16:
        _to_fp16(output_path)

    logger.info(f"ONNX model written to {output_path}")

    # Report the real input/output names so config.pbtxt can be validated.
    try:
        import onnx

        graph = onnx.load(str(output_path)).graph
        inputs = [i.name for i in graph.input]
        outputs = [o.name for o in graph.output]
        logger.info(f"ONNX inputs : {inputs}")
        logger.info(f"ONNX outputs: {outputs}")
        logger.info(
            "Make sure model_repository/classifier/config.pbtxt lists exactly these "
            "inputs/outputs."
        )
    except ImportError:
        logger.warning("`onnx` not installed; skipping input/output inspection.")


if __name__ == "__main__":
    main()

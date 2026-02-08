"""
ONNX export utilities for optimized inference.
"""

from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from loguru import logger

from src.config import settings


def export_to_onnx(
    model_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    opset_version: int = 14,
    optimize: bool = True,
    fp16: bool = True,
) -> Path:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model_path: Path to saved PyTorch model
        output_path: Path for ONNX output file
        opset_version: ONNX opset version
        optimize: Apply ONNX graph optimizations
        fp16: Convert to FP16 for faster inference
        
    Returns:
        Path to exported ONNX model
    """
    if model_path is None:
        model_path = settings.models_dir
    
    if output_path is None:
        suffix = "_fp16.onnx" if fp16 else ".onnx"
        output_path = model_path / f"model{suffix}"
    
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model.eval()
    
    # Create dummy input
    dummy_text = "Esempio di testo per l'export ONNX"
    dummy_input = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=settings.model.max_length,
    )
    
    # Define dynamic axes for variable batch size and sequence length
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "token_type_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    }
    
    # Export
    logger.info(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"], dummy_input["token_type_ids"]),
        str(output_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    if optimize:
        output_path = _optimize_onnx(output_path, fp16)
    
    logger.info(f"ONNX model saved to {output_path}")
    return output_path


def _optimize_onnx(model_path: Path, fp16: bool = True) -> Path:
    """Apply ONNX Runtime optimizations."""
    try:
        import onnx
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.fusion_options import FusionOptions
    except ImportError:
        logger.warning("ONNX optimization libraries not available, skipping optimization")
        return model_path
    
    logger.info("Applying ONNX optimizations...")
    
    # Optimize using architecture from config
    opt_options = FusionOptions(settings.model.architecture.model_type)
    optimized_model = optimizer.optimize_model(
        str(model_path),
        model_type=settings.model.architecture.model_type,
        num_heads=settings.model.architecture.num_heads,
        hidden_size=settings.model.architecture.hidden_size,
        optimization_options=opt_options,
    )
    
    if fp16:
        logger.info("Converting to FP16...")
        optimized_model.convert_float_to_float16(keep_io_types=True)
    
    # Save optimized model
    output_path = model_path.parent / f"model_optimized{'_fp16' if fp16 else ''}.onnx"
    optimized_model.save_model_to_file(str(output_path))
    
    # Get size comparison
    original_size = model_path.stat().st_size / (1024 * 1024)
    optimized_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Model size: {original_size:.1f}MB → {optimized_size:.1f}MB")
    
    return output_path

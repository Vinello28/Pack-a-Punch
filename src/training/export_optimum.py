"""
ONNX export with Hugging Face Optimum and kernel fusion.

This module provides optimized ONNX export using Optimum's ORTOptimizer
which fuses attention layers into efficient CUDA kernels, eliminating
Memcpy operations between CPU and GPU.
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from src.config import settings


def export_with_optimum(
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    fp16: bool = True,
    device: str = "cuda",
) -> Path:
    """
    Export PyTorch model to ONNX using Hugging Face Optimum with kernel fusion.
    
    This method creates an optimized ONNX model with fused attention kernels
    that run efficiently on GPU without excessive Memcpy operations.
    
    Args:
        model_path: Path to saved PyTorch model directory
        output_dir: Directory for optimized ONNX output
        fp16: Use FP16 precision for faster inference
        device: Target device ("cuda" or "cpu")
        
    Returns:
        Path to the optimized ONNX model file
    """
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer
        from optimum.onnxruntime.configuration import OptimizationConfig
    except ImportError as e:
        raise ImportError(
            "Optimum is required for this export method. "
            "Install with: pip install 'optimum[onnxruntime-gpu]'"
        ) from e
    
    if model_path is None:
        model_path = settings.models_dir
    model_path = Path(model_path)
    
    if output_dir is None:
        output_dir = model_path / "optimum_export"
    output_dir = Path(output_dir)
    
    logger.info(f"Loading model from {model_path} with Optimum (export=True)")
    
    # Step 1: Load and export model using Optimum
    # export=True forces proper ONNX conversion with Optimum's optimizations
    model = ORTModelForSequenceClassification.from_pretrained(
        model_path,
        export=True,
    )
    
    # Save initial export
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(temp_dir)
    logger.info(f"Initial ONNX export saved to {temp_dir}")
    
    # Step 2: Apply kernel fusion optimizations
    # This is the CRUCIAL step that eliminates Memcpy operations
    logger.info("Applying kernel fusion optimizations (O2 level)...")
    
    optimizer = ORTOptimizer.from_pretrained(temp_dir)
    
    # Configure optimization based on device and precision
    # IMPORTANT: Force model_type="bert" to enable BERT attention fusion for ModernBERT
    # ModernBERT is not recognized by default, but uses standard BERT attention patterns
    if device == "cuda":
        optimization_config = OptimizationConfig(
            optimization_level=2,  # O2 level
            optimize_for_gpu=True,
            fp16=fp16,
            enable_transformers_specific_optimizations=True,
        )
    else:
        optimization_config = OptimizationConfig(
            optimization_level=1,  # O1 for CPU
            optimize_for_gpu=False,
            fp16=False,
        )
    
    # Final output directory
    final_output = output_dir / f"model_optimum{'_fp16' if fp16 else ''}"
    final_output.mkdir(parents=True, exist_ok=True)
    
    optimizer.optimize(
        save_dir=str(final_output),
        optimization_config=optimization_config,
    )
    
    # Find the generated ONNX file
    onnx_files = list(final_output.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX file found in {final_output}")
    
    onnx_path = onnx_files[0]
    
    # Copy to models directory with standard name for engine.py compatibility
    import shutil
    final_model = model_path / f"model_optimized{'_fp16' if fp16 else ''}.onnx"
    shutil.copy(onnx_path, final_model)
    
    # Also copy tokenizer files
    for tokenizer_file in model_path.glob("tokenizer*"):
        shutil.copy(tokenizer_file, final_output / tokenizer_file.name)
    for config_file in model_path.glob("*.json"):
        if not (final_output / config_file.name).exists():
            shutil.copy(config_file, final_output / config_file.name)
    
    # Log size info
    size_mb = final_model.stat().st_size / (1024 * 1024)
    logger.info(f"Optimized model size: {size_mb:.1f} MB")
    
    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    logger.info(f"✓ Optimized ONNX model saved to: {final_model}")
    logger.info(f"✓ Full export directory: {final_output}")
    
    return final_model


def verify_kernel_fusion(model_path: Path) -> dict:
    """
    Verify that kernel fusion was applied by counting graph nodes.
    
    A well-optimized model should have significantly fewer nodes
    than a standard export.
    
    Args:
        model_path: Path to ONNX model file
        
    Returns:
        Dict with node statistics
    """
    try:
        import onnx
    except ImportError:
        logger.warning("ONNX not installed, cannot verify fusion")
        return {}
    
    model = onnx.load(str(model_path))
    
    # Count nodes by type
    node_counts = {}
    for node in model.graph.node:
        op_type = node.op_type
        node_counts[op_type] = node_counts.get(op_type, 0) + 1
    
    total_nodes = len(model.graph.node)
    
    # Check for fused attention
    has_fused_attention = any(
        "Attention" in op or "FusedMatMul" in op or "FusedGemm" in op
        for op in node_counts.keys()
    )
    
    logger.info(f"Model nodes: {total_nodes}")
    logger.info(f"Fused attention detected: {has_fused_attention}")
    
    return {
        "total_nodes": total_nodes,
        "node_types": node_counts,
        "has_fused_attention": has_fused_attention,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export model with Optimum")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--verify", action="store_true", help="Verify fusion after export")
    
    args = parser.parse_args()
    
    result = export_with_optimum(
        model_path=args.model_path,
        output_dir=args.output_dir,
        fp16=not args.no_fp16,
        device=args.device,
    )
    
    if args.verify:
        verify_kernel_fusion(result)

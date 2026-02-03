#!/usr/bin/env python3
"""
Training script for Pack-a-Punch classifier.

Usage:
    python scripts/train.py --data-source txt
    python scripts/train.py --data-source distillation --teacher-url http://localhost:1234
    python scripts/train.py --data-source jsonl --export-onnx
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from loguru import logger

from src.config import settings
from src.training.trainer import Trainer
from src.training.distillation import run_distillation
from src.training.export_onnx import export_to_onnx


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pack-a-Punch classifier")
    
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["auto", "txt", "jsonl", "distillation"],
        default="auto",
        help="Data source for training",
    )
    
    parser.add_argument(
        "--teacher-url",
        type=str,
        default=settings.distillation.teacher_url,
        help="LLM Teacher API URL (for distillation)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.training.batch_size,
        help="Training batch size",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=settings.training.num_epochs,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=settings.training.learning_rate,
        help="Learning rate",
    )
    
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export to ONNX after training",
    )
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 training",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.models_dir,
        help="Output directory for model",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Pack-a-Punch Training")
    logger.info("=" * 60)
    
    # Handle distillation
    if args.data_source == "distillation":
        logger.info("Running distillation pipeline...")
        settings.distillation.teacher_url = args.teacher_url
        
        try:
            run_distillation()
            args.data_source = "distilled"
        except Exception as e:
            logger.error(f"Distillation failed: {e}")
            return 1
    
    # Train
    logger.info("Starting training...")
    trainer = Trainer(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        fp16=args.fp16,
        output_dir=args.output_dir,
    )
    
    try:
        model_path = trainer.train(data_source=args.data_source)
        logger.info(f"Training complete. Model saved to: {model_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    # Export ONNX
    if args.export_onnx:
        logger.info("Exporting to ONNX...")
        try:
            onnx_path = export_to_onnx(model_path=model_path, fp16=args.fp16)
            logger.info(f"ONNX model saved to: {onnx_path}")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return 1
    
    logger.info("=" * 60)
    logger.info("Training pipeline complete!")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

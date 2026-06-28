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
import torch

torch.set_float32_matmul_precision('high')

from src.config import settings
from src.training.trainer import Trainer
from src.training.dataset import load_dataset_from_csv
from src.training.distillation import run_distillation

# open-data-analytics project root (scripts/ -> inference-usage -> src -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[3]
TRACEABILITY_DIR = REPO_ROOT / "data" / "traceability" / "training"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pack-a-Punch classifier")
    
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["auto", "txt", "jsonl", "distillation", "csv"],
        default="auto",
        help="Data source for training",
    )

    parser.add_argument(
        "--csv-path",
        type=Path,
        default=TRACEABILITY_DIR / "train_traceability.csv",
        help="Training CSV (TITOLO_PROGETTO, DESCRIZIONE_PROGETTO, label) for --data-source csv",
    )

    parser.add_argument(
        "--test-csv-path",
        type=Path,
        default=TRACEABILITY_DIR / "test_traceability.csv",
        help="Held-out test CSV evaluated after training (for --data-source csv)",
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
    
    parser.add_argument(
        "--kfold",
        action="store_true",
        default=settings.training.kfold_enabled,
        help="Use Stratified K-Fold Cross Validation instead of simple split",
    )
    
    parser.add_argument(
        "--kfold-splits",
        type=int,
        default=settings.training.kfold_splits,
        help="Number of folds for K-Fold CV (default: 5)",
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
    
    # The traceability CSV ships its own held-out test set, so we use a single
    # train/eval split (not K-Fold) and evaluate on that test set after training.
    texts = labels = None
    if args.data_source == "csv":
        logger.info(f"Loading traceability training CSV: {args.csv_path}")
        texts, labels = load_dataset_from_csv(args.csv_path)
        args.kfold = False

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
        if args.kfold:
            logger.info(f"Using {args.kfold_splits}-Fold Stratified Cross Validation")
            model_path = trainer.train_kfold(
                n_splits=args.kfold_splits,
                data_source=args.data_source,
            )
        elif texts is not None:
            model_path = trainer.train(texts=texts, labels=labels)
        else:
            model_path = trainer.train(data_source=args.data_source)
        logger.info(f"Training complete. Model saved to: {model_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

    # Evaluate on the held-out traceability test set
    if args.data_source == "csv" and args.test_csv_path.exists():
        logger.info(f"Evaluating on held-out test set: {args.test_csv_path}")
        test_texts, test_labels = load_dataset_from_csv(args.test_csv_path)
        test_loader = trainer._create_dataloader(
            test_texts, test_labels, shuffle=False, batch_size=trainer.batch_size * 2
        )
        m = trainer._evaluate(test_loader)
        logger.info(
            f"HELD-OUT TEST  acc={m['accuracy']:.4f}  P={m['precision']:.4f}  "
            f"R={m['recall']:.4f}  F1={m['f1']:.4f}  loss={m['loss']:.4f}"
        )
    

    
    logger.info("=" * 60)
    logger.info("Training pipeline complete!")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

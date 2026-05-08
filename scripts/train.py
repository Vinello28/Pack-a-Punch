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
from src.training.export_optimum import export_with_optimum


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
        help="Export to ONNX after training (standard method)",
    )
    
    parser.add_argument(
        "--export-optimum",
        action="store_true",
        help="Export to ONNX using Optimum with kernel fusion (recommended for GPU)",
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
        else:
            model_path = trainer.train(data_source=args.data_source)
        logger.info(f"Training complete. Model saved to: {model_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    # Export ONNX
    onnx_path = None
    if args.export_optimum:
        logger.info("Exporting to ONNX with Optimum (kernel fusion)...")
        try:
            onnx_path = export_with_optimum(model_path=model_path, fp16=args.fp16)
            logger.info(f"Optimized ONNX model saved to: {onnx_path}")
        except Exception as e:
            logger.error(f"Optimum export failed: {e}")
            return 1
    elif args.export_onnx:
        logger.info("Exporting to ONNX (standard method)...")
        try:
            onnx_path = export_to_onnx(model_path=model_path, fp16=args.fp16)
            logger.info(f"ONNX model saved to: {onnx_path}")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return 1
    
    # Evaluation on Test set
    if model_path:
        logger.info("=" * 60)
        logger.info("Evaluating PyTorch model on Test set...")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from src.training.dataset import load_dataset_from_txt
            from sklearn.metrics import classification_report
            import time
            
            test_dir = settings.data_dir / "Test"
            if test_dir.exists():
                logger.info(f"Loading test data from {test_dir}...")
                test_texts, test_labels = load_dataset_from_txt(test_dir)
                
                logger.info("Loading PyTorch model and tokenizer...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                model.to(device)
                model.eval()
                
                logger.info("Running predictions...")
                start_time = time.time()
                
                batch_size = 64
                pred_labels = []
                
                for i in range(0, len(test_texts), batch_size):
                    batch_texts = test_texts[i:i + batch_size]
                    inputs = tokenizer(
                        batch_texts, 
                        padding=True, 
                        truncation=True, 
                        max_length=settings.model.max_length, 
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=-1).cpu().tolist()
                        pred_labels.extend(preds)
                
                logger.info(f"Inference time: {time.time() - start_time:.2f}s")
                
                # Print metrics
                target_names = [settings.model.label_map[0], settings.model.label_map[1]]
                report = classification_report(
                    test_labels, 
                    pred_labels, 
                    target_names=target_names
                )
                logger.info("Classification Report:\n" + report)
            else:
                logger.warning(f"Test directory not found at {test_dir}. Skipping evaluation.")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")

    logger.info("=" * 60)
    logger.info("Training pipeline complete!")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

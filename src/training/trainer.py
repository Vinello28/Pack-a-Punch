"""
Fine-tuning trainer for BERT text classification.
Supports both simple train/eval split and Stratified K-Fold Cross Validation.
"""

from pathlib import Path
from typing import Optional
import copy
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from loguru import logger
from tqdm import tqdm

from src.config import settings
from .dataset import TextClassificationDataset, load_dataset


class Trainer:
    """Fine-tuning trainer with early stopping, checkpoint saving, and K-Fold CV."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        output_dir: Optional[Path] = None,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        fp16: bool = True,
        early_stopping_patience: int = 3,
        eval_split: float = 0.1,
    ):
        self.model_name = model_name or settings.model.name
        self.output_dir = output_dir or settings.models_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16 and torch.cuda.is_available()
        self.early_stopping_patience = early_stopping_patience
        self.eval_split = eval_split
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer (shared across folds)
        logger.info(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize or reset the model to a fresh pretrained state."""
        logger.info(f"Loading model: {self.model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=settings.model.num_labels,
        ).to(self.device)
        
        if self.fp16:
            self.scaler = torch.amp.GradScaler('cuda')
            logger.info("Using mixed precision (FP16)")
    
    def _create_dataloader(
        self,
        texts: list[str],
        labels: list[int],
        shuffle: bool = True,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        """Create a DataLoader from texts and labels."""
        dataset = TextClassificationDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=settings.model.max_length,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
        )
    
    def _prepare_data(
        self,
        texts: list[str],
        labels: list[int],
    ) -> tuple[DataLoader, DataLoader]:
        """Prepare train and eval dataloaders with random split."""
        dataset = TextClassificationDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=settings.model.max_length,
        )
        
        # Split into train/eval
        eval_size = int(len(dataset) * self.eval_split)
        train_size = len(dataset) - eval_size
        
        train_dataset, eval_dataset = random_split(
            dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(42),
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        
        logger.info(f"Train size: {train_size}, Eval size: {eval_size}")
        return train_loader, eval_loader
    
    def _evaluate(self, eval_loader: DataLoader) -> dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = {
            "loss": total_loss / len(eval_loader),
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="binary"),
            "precision": precision_score(all_labels, all_preds, average="binary"),
            "recall": recall_score(all_labels, all_preds, average="binary"),
        }
        
        return metrics
    
    def _run_training_loop(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        fold_label: str = "",
    ) -> dict[str, float]:
        """
        Run a single training loop (used by both train and train_kfold).
        
        Returns:
            Best metrics achieved during training.
        """
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Scheduler
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Training loop
        best_f1 = 0.0
        best_metrics = {}
        patience_counter = 0
        prefix = f"[{fold_label}] " if fold_label else ""
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            
            progress_bar = tqdm(
                train_loader,
                desc=f"{prefix}Epoch {epoch + 1}/{self.num_epochs}",
            )
            
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_batch = batch["labels"].to(self.device)
                
                optimizer.zero_grad()
                
                if self.fp16:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels_batch,
                        )
                        loss = outputs.loss
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels_batch,
                    )
                    loss = outputs.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    optimizer.step()
                
                scheduler.step()
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_loss = total_loss / len(train_loader)
            
            # Evaluate
            metrics = self._evaluate(eval_loader)
            logger.info(
                f"{prefix}Epoch {epoch + 1} - "
                f"Train Loss: {avg_loss:.4f}, "
                f"Eval Loss: {metrics['loss']:.4f}, "
                f"Acc: {metrics['accuracy']:.4f}, "
                f"P: {metrics['precision']:.4f}, "
                f"R: {metrics['recall']:.4f}, "
                f"F1: {metrics['f1']:.4f}"
            )
            
            # Early stopping
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_metrics = metrics.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"{prefix}Early stopping at epoch {epoch + 1}")
                    break
        
        return best_metrics
    
    def train(
        self,
        texts: Optional[list[str]] = None,
        labels: Optional[list[int]] = None,
        data_source: str = "auto",
    ) -> Path:
        """
        Run training loop with simple train/eval split.
        
        Args:
            texts: Training texts (or load from data_source)
            labels: Training labels
            data_source: One of "auto", "txt", "jsonl", "distilled"
            
        Returns:
            Path to saved model
        """
        if texts is None or labels is None:
            texts, labels = load_dataset(source=data_source)
        
        train_loader, eval_loader = self._prepare_data(texts, labels)
        
        best_metrics = self._run_training_loop(train_loader, eval_loader)
        
        if best_metrics:
            logger.info(f"Best validation F1: {best_metrics['f1']:.4f}")
            self._save_model()
        
        return self.output_dir
    
    def train_kfold(
        self,
        n_splits: int = 5,
        texts: Optional[list[str]] = None,
        labels: Optional[list[int]] = None,
        data_source: str = "auto",
    ) -> Path:
        """
        Run Stratified K-Fold Cross Validation training.
        
        For each fold:
        1. Reinitialize model from pretrained weights
        2. Train on K-1 folds
        3. Evaluate on held-out fold
        
        After all folds, retrain on ALL data and save the final model.
        
        Args:
            n_splits: Number of folds (default: 5)
            texts: Training texts (or load from data_source)
            labels: Training labels
            data_source: One of "auto", "txt", "jsonl", "distilled"
            
        Returns:
            Path to saved model
        """
        if texts is None or labels is None:
            texts, labels = load_dataset(source=data_source)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        all_fold_metrics: list[dict[str, float]] = []
        
        logger.info("=" * 60)
        logger.info(f"Starting {n_splits}-Fold Stratified Cross Validation")
        logger.info(f"Total samples: {len(texts)} (AI: {sum(labels)}, NON_AI: {len(labels) - sum(labels)})")
        logger.info("=" * 60)
        
        texts_arr = np.array(texts)
        labels_arr = np.array(labels)
        
        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(texts_arr, labels_arr)):
            fold_label = f"Fold {fold_idx + 1}/{n_splits}"
            logger.info("-" * 60)
            logger.info(f"{fold_label} starting...")
            
            # Extract fold data
            train_texts = texts_arr[train_indices].tolist()
            train_labels = labels_arr[train_indices].tolist()
            val_texts = texts_arr[val_indices].tolist()
            val_labels = labels_arr[val_indices].tolist()
            
            train_ai = sum(train_labels)
            val_ai = sum(val_labels)
            logger.info(
                f"{fold_label} - Train: {len(train_texts)} "
                f"(AI: {train_ai}, NON_AI: {len(train_texts) - train_ai}) | "
                f"Val: {len(val_texts)} "
                f"(AI: {val_ai}, NON_AI: {len(val_texts) - val_ai})"
            )
            
            # Reinitialize model for each fold (fresh pretrained weights)
            self._init_model()
            
            # Create dataloaders
            train_loader = self._create_dataloader(
                train_texts, train_labels, shuffle=True
            )
            val_loader = self._create_dataloader(
                val_texts, val_labels, shuffle=False,
                batch_size=self.batch_size * 2,
            )
            
            # Train this fold
            fold_metrics = self._run_training_loop(
                train_loader, val_loader, fold_label=fold_label
            )
            all_fold_metrics.append(fold_metrics)
            
            logger.info(
                f"{fold_label} RESULT - "
                f"Acc: {fold_metrics['accuracy']:.4f}, "
                f"P: {fold_metrics['precision']:.4f}, "
                f"R: {fold_metrics['recall']:.4f}, "
                f"F1: {fold_metrics['f1']:.4f}"
            )
            
            # Free GPU memory between folds
            del train_loader, val_loader
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # === Aggregate results ===
        logger.info("=" * 60)
        logger.info(f"{n_splits}-FOLD CROSS VALIDATION RESULTS")
        logger.info("=" * 60)
        
        metric_names = ["accuracy", "precision", "recall", "f1", "loss"]
        cv_results = {}
        
        for metric in metric_names:
            values = [m[metric] for m in all_fold_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv_results[metric] = {"mean": float(mean_val), "std": float(std_val), "values": [float(v) for v in values]}
            logger.info(f"  {metric:>12s}: {mean_val:.4f} ± {std_val:.4f}  (per fold: {[f'{v:.4f}' for v in values]})")
        
        # Save CV results to JSON
        cv_report_path = self.output_dir / "cv_results.json"
        cv_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cv_report_path, "w") as f:
            json.dump({
                "n_splits": n_splits,
                "total_samples": len(texts),
                "label_distribution": {"AI": int(sum(labels)), "NON_AI": int(len(labels) - sum(labels))},
                "metrics": cv_results,
                "fold_details": all_fold_metrics,
            }, f, indent=2)
        logger.info(f"CV results saved to: {cv_report_path}")
        
        # === Final training on ALL data ===
        logger.info("=" * 60)
        logger.info("Retraining final model on ALL data...")
        logger.info("=" * 60)
        
        self._init_model()
        
        # For final model we use a small held-out for early stopping only
        # (this is standard practice: CV gives the estimate, retrain on all gives the model)
        train_loader, eval_loader = self._prepare_data(texts, labels)
        final_metrics = self._run_training_loop(
            train_loader, eval_loader, fold_label="Final"
        )
        
        self._save_model()
        logger.info(f"Final model saved to: {self.output_dir}")
        logger.info(f"Final model best F1: {final_metrics.get('f1', 0):.4f}")
        
        return self.output_dir
    
    def _save_model(self):
        """Save model and tokenizer."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Model saved to {self.output_dir}")

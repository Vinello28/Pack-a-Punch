"""
Fine-tuning trainer for BERT text classification.
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from loguru import logger
from tqdm import tqdm

from src.config import settings
from .dataset import TextClassificationDataset, load_dataset


class Trainer:
    """Fine-tuning trainer with early stopping and checkpoint saving."""
    
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
        
        # Initialize model and tokenizer
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=settings.model.num_labels,
        ).to(self.device)
        
        if self.fp16:
            self.scaler = torch.amp.GradScaler('cuda')
            logger.info("Using mixed precision (FP16)")
    
    def _prepare_data(
        self,
        texts: list[str],
        labels: list[int],
    ) -> tuple[DataLoader, DataLoader]:
        """Prepare train and eval dataloaders."""
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
    
    def train(
        self,
        texts: Optional[list[str]] = None,
        labels: Optional[list[int]] = None,
        data_source: str = "auto",
    ) -> Path:
        """
        Run training loop.
        
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
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
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
                f"Epoch {epoch + 1} - "
                f"Train Loss: {avg_loss:.4f}, "
                f"Eval Loss: {metrics['loss']:.4f}, "
                f"Acc: {metrics['accuracy']:.4f}, "
                f"F1: {metrics['f1']:.4f}"
            )
            
            # Early stopping
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                patience_counter = 0
                self._save_model()
                logger.info(f"New best F1: {best_f1:.4f} - Model saved")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        return self.output_dir
    
    def _save_model(self):
        """Save model and tokenizer."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Model saved to {self.output_dir}")

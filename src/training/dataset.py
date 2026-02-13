"""
Dataset loading utilities for text classification.

Supports:
1. TXT files organized in label directories (data/ai/, data/non_ai/)
2. JSONL files with {"text": "...", "label": 0|1} format
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from loguru import logger

from src.config import settings


class TextClassificationDataset(Dataset):
    """PyTorch Dataset for text classification."""
    
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_dataset_from_txt(
    data_dir: Optional[Path] = None,
    ai_subdir: str = "ai",
    non_ai_subdir: str = "non_ai",
    in_domain_only: bool = True,
) -> tuple[list[str], list[int]]:
    """
    Load dataset from TXT files organized in label directories.
    
    Expected structure:
    data_dir/
    ├── ai/
    │   ├── tbc_0.txt
    │   └── tbc_1.txt
    └── non_ai/
        ├── tbc_2.txt
        └── tbc_3.txt
    
    Args:
        data_dir: Base data directory (default: settings.data_dir)
        ai_subdir: Subdirectory name for texts about AI
        non_ai_subdir: Subdirectory name for texts about other topics
        in_domain_only: If True, only load tbc_* files (verified in-domain data)
        
    Returns:
        Tuple of (texts, labels) where label 1 = AI, 0 = NON_AI
    """
    if data_dir is None:
        data_dir = settings.data_dir
    
    texts = []
    labels = []
    
    glob_pattern = "tbc_*.txt" if in_domain_only else "*.txt"
    if in_domain_only:
        logger.info("In-domain mode: loading only tbc_* files")
    
    # Load AI texts (label = 1)
    ai_dir = data_dir / ai_subdir
    if ai_dir.exists():
        for txt_file in sorted(ai_dir.glob(glob_pattern)):
            content = txt_file.read_text(encoding="utf-8").strip()
            if content:
                texts.append(content)
                labels.append(1)
        logger.info(f"Loaded {len([l for l in labels if l == 1])} AI samples from {ai_dir}")
    else:
        logger.warning(f"AI directory not found: {ai_dir}")
    
    # Load NON_AI texts (label = 0)
    non_ai_dir = data_dir / non_ai_subdir
    if non_ai_dir.exists():
        for txt_file in sorted(non_ai_dir.glob(glob_pattern)):
            content = txt_file.read_text(encoding="utf-8").strip()
            if content:
                texts.append(content)
                labels.append(0)
        logger.info(f"Loaded {len([l for l in labels if l == 0])} NON_AI samples from {non_ai_dir}")
    else:
        logger.warning(f"NON_AI directory not found: {non_ai_dir}")
    
    if not texts:
        raise ValueError(f"No training data found in {data_dir}")
    
    logger.info(f"Total dataset size: {len(texts)} samples")
    return texts, labels


def load_dataset_from_jsonl(
    file_path: Optional[Path] = None,
) -> tuple[list[str], list[int]]:
    """
    Load dataset from JSONL file.
    
    Expected format (one JSON object per line):
    {"text": "example text", "label": 1}
    {"text": "another text", "label": 0}
    
    Args:
        file_path: Path to JSONL file (default: settings.data_dir / "train.jsonl")
        
    Returns:
        Tuple of (texts, labels)
    """
    if file_path is None:
        file_path = settings.data_dir / "train.jsonl"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    texts = []
    labels = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
                text = item.get("text", "").strip()
                label = item.get("label")
                
                if not text:
                    logger.warning(f"Empty text at line {line_num}, skipping")
                    continue
                    
                if label not in (0, 1):
                    # Try to convert string labels
                    if isinstance(label, str):
                        label = 1 if label.upper() == "AI" else 0
                    else:
                        logger.warning(f"Invalid label at line {line_num}: {label}")
                        continue
                
                texts.append(text)
                labels.append(label)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON at line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(texts)} samples from {file_path}")
    logger.info(f"Label distribution: AI={sum(labels)}, NON_AI={len(labels) - sum(labels)}")
    
    return texts, labels


def load_dataset(
    source: str = "auto",
    data_dir: Optional[Path] = None,
) -> tuple[list[str], list[int]]:
    """
    Auto-detect and load dataset from available sources.
    
    Args:
        source: One of "auto", "txt", "jsonl", "distilled"
        data_dir: Base data directory
        
    Returns:
        Tuple of (texts, labels)
    """
    if data_dir is None:
        data_dir = settings.data_dir
    
    if source == "txt":
        return load_dataset_from_txt(data_dir)
    
    if source == "jsonl":
        return load_dataset_from_jsonl(data_dir / "train.jsonl")
    
    if source == "distilled":
        return load_dataset_from_jsonl(data_dir / "distilled.jsonl")
    
    # Auto-detect
    if (data_dir / "distilled.jsonl").exists():
        logger.info("Auto-detected: distilled.jsonl")
        return load_dataset_from_jsonl(data_dir / "distilled.jsonl")
    
    if (data_dir / "train.jsonl").exists():
        logger.info("Auto-detected: train.jsonl")
        return load_dataset_from_jsonl(data_dir / "train.jsonl")
    
    if (data_dir / "ai").exists() or (data_dir / "non_ai").exists():
        logger.info("Auto-detected: TXT directories")
        return load_dataset_from_txt(data_dir)
    
    raise ValueError(
        f"No dataset found in {data_dir}. "
        "Expected: train.jsonl, distilled.jsonl, or ai/non_ai directories"
    )

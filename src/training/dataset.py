"""
Dataset loading utilities for text classification.

Supports:
1. TXT files organized in label directories (one subdirectory per class)
2. JSONL files with {"text": "...", "label": 0|1|...} format
3. CSV files with "description" (or legacy "Descrizione") and "Label" columns
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from loguru import logger

from src.config import settings


def _slugify(name: str) -> str:
    """Convert a label name to a directory slug (lowercase, underscores)."""
    slug = name.lower()
    slug = slug.replace("&", "").replace(",", "")
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return re.sub(r"_+", "_", slug)


def _build_slug_to_id() -> dict[str, int]:
    """Build mapping from directory slug to label index."""
    return {_slugify(name): idx for idx, name in settings.model.label_map.items()}


def _build_name_to_id() -> dict[str, int]:
    """Build mapping from label name to label index (case-insensitive)."""
    return {name.lower(): idx for idx, name in settings.model.label_map.items()}


class TextClassificationDataset(Dataset):
    """PyTorch Dataset for text classification."""
    
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 3072,
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
    in_domain_only: bool = False,
) -> tuple[list[str], list[int]]:
    """
    Load dataset from TXT files organized in label directories.

    Each subdirectory name is matched against the slugified label names
    from settings.model.label_map.

    Args:
        data_dir: Base data directory (default: settings.data_dir)
        in_domain_only: If True, only load tbc_* files (verified in-domain data)

    Returns:
        Tuple of (texts, labels)
    """
    if data_dir is None:
        data_dir = settings.data_dir

    slug_to_id = _build_slug_to_id()
    texts = []
    labels = []

    glob_pattern = "tbc_*.txt" if in_domain_only else "*.txt"
    if in_domain_only:
        logger.info("In-domain mode: loading only tbc_* files")

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        label_id = slug_to_id.get(subdir.name)
        if label_id is None:
            continue
        count = 0
        for txt_file in sorted(subdir.glob(glob_pattern)):
            content = txt_file.read_text(encoding="utf-8").strip()
            if content:
                texts.append(content)
                labels.append(label_id)
                count += 1
        label_name = settings.model.label_map[label_id]
        logger.info(f"Loaded {count} '{label_name}' samples from {subdir}")

    if not texts:
        raise ValueError(f"No training data found in {data_dir}")

    dist = Counter(labels)
    dist_str = ", ".join(f"{settings.model.label_map[k]}: {v}" for k, v in sorted(dist.items()))
    logger.info(f"Total dataset size: {len(texts)} samples ({dist_str})")
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
                    
                valid_ids = set(range(settings.model.num_labels))
                if isinstance(label, int) and label in valid_ids:
                    pass  # already a valid integer label
                elif isinstance(label, str):
                    name_to_id = _build_name_to_id()
                    resolved = name_to_id.get(label.lower())
                    if resolved is None:
                        logger.warning(f"Unknown label at line {line_num}: {label}")
                        continue
                    label = resolved
                else:
                    logger.warning(f"Invalid label at line {line_num}: {label}")
                    continue
                
                texts.append(text)
                labels.append(label)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON at line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(texts)} samples from {file_path}")
    dist = Counter(labels)
    dist_str = ", ".join(f"{settings.model.label_map[k]}: {v}" for k, v in sorted(dist.items()))
    logger.info(f"Label distribution: {dist_str}")
    
    return texts, labels


def load_dataset_from_csv(
    file_path: Path,
) -> tuple[list[str], list[int]]:
    """
    Load dataset from a CSV file with "description" (or legacy "Descrizione")
    and "Label" columns.

    Rows whose label does not match any entry in settings.model.label_map
    are skipped with a warning (handles malformed/noisy rows).

    Args:
        file_path: Path to the CSV file.

    Returns:
        Tuple of (texts, labels)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    name_to_id = _build_name_to_id()
    texts = []
    labels = []
    skipped = 0

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        text_col = "description" if reader.fieldnames and "description" in reader.fieldnames else "Descrizione"
        for row_num, row in enumerate(reader, 2):  # row 1 is header
            text = (row.get(text_col) or "").strip()
            label_str = (row.get("Label") or "").strip()

            if not text:
                logger.warning(f"Empty text at row {row_num}, skipping")
                skipped += 1
                continue

            label_id = name_to_id.get(label_str.lower())
            if label_id is None:
                skipped += 1
                continue

            texts.append(text)
            labels.append(label_id)

    if skipped:
        logger.warning(f"Skipped {skipped} rows with empty text or unrecognised labels")

    if not texts:
        raise ValueError(f"No valid samples found in {file_path}")

    dist = Counter(labels)
    dist_str = ", ".join(f"{settings.model.label_map[k]}: {v}" for k, v in sorted(dist.items()))
    logger.info(f"Loaded {len(texts)} samples from {file_path} ({dist_str})")
    return texts, labels


def load_dataset(
    source: str = "auto",
    data_dir: Optional[Path] = None,
    csv_path: Optional[Path] = None,
) -> tuple[list[str], list[int]]:
    """
    Auto-detect and load dataset from available sources.

    Args:
        source: One of "auto", "txt", "jsonl", "distilled", "csv"
        data_dir: Base data directory
        csv_path: Path to CSV file (required when source="csv")

    Returns:
        Tuple of (texts, labels)
    """
    if data_dir is None:
        data_dir = settings.data_dir
    
    if source == "txt":
        return load_dataset_from_txt(data_dir)

    if source == "csv":
        if csv_path is None:
            raise ValueError("csv_path is required when source='csv'")
        return load_dataset_from_csv(csv_path)

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
    
    # Check if any subdirectory matches a label slug
    slug_to_id = _build_slug_to_id()
    has_label_dirs = any(
        subdir.is_dir() and subdir.name in slug_to_id
        for subdir in data_dir.iterdir()
        if subdir.is_dir()
    )
    if has_label_dirs:
        logger.info("Auto-detected: TXT directories")
        return load_dataset_from_txt(data_dir)

    raise ValueError(
        f"No dataset found in {data_dir}. "
        "Expected: train.jsonl, distilled.jsonl, or label subdirectories"
    )

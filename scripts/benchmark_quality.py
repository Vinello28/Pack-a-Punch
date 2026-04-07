#!/usr/bin/env python3
"""
Benchmark Quality Script

Measures classification quality (F1, Accuracy, Precision, Recall) using a labeled test set.
Iterates through class subdirectories, sends requests to the inference API,
and computes metrics comparing predictions against ground truth.
"""

import sys
import argparse
import asyncio
import httpx
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from loguru import logger
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.training.dataset import _slugify


async def classify_batch(client: httpx.AsyncClient, url: str, texts: list[str]) -> list[dict]:
    try:
        response = await client.post(f"{url}/classify", json={"texts": texts})
        response.raise_for_status()
        return response.json()["predictions"]
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return []


async def run_benchmark(test_dir: Path, url: str, batch_size: int = 10):
    # Build mapping from label name to index
    name_to_id = {name: idx for idx, name in settings.model.label_map.items()}
    slug_to_id = {_slugify(name): idx for idx, name in settings.model.label_map.items()}

    # 1. Load Data from class subdirectories
    samples = []

    for subdir in sorted(test_dir.iterdir()):
        if not subdir.is_dir():
            continue
        label_id = slug_to_id.get(subdir.name)
        if label_id is None:
            continue
        label_name = settings.model.label_map[label_id]
        count = 0
        for p in sorted(subdir.glob("*.txt")):
            try:
                text = p.read_text(encoding="utf-8").strip()
                if text:
                    samples.append({"text": text, "true_label": label_id, "filename": p.name})
                    count += 1
            except Exception as e:
                logger.warning(f"Could not read {p}: {e}")
        logger.info(f"Loaded {count} '{label_name}' samples from {subdir}")

    if not samples:
        logger.error("No samples found!")
        sys.exit(1)

    from collections import Counter

    dist = Counter(s["true_label"] for s in samples)
    dist_str = ", ".join(f"{settings.model.label_map[k]}: {v}" for k, v in sorted(dist.items()))
    logger.info(f"Total samples: {len(samples)} ({dist_str})")

    # 2. Run Inference
    y_true = []
    y_pred = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Check health
        try:
            resp = await client.get(f"{url}/health")
            resp.raise_for_status()
            logger.info(f"Service status: {resp.json().get('status')}")
        except Exception as e:
            logger.error(f"Service check failed at {url}: {e}")
            logger.error("Is the inference service running? (docker compose up ...)")
            sys.exit(1)

        # Process in batches
        all_texts = [s["text"] for s in samples]
        chunks = [all_texts[i : i + batch_size] for i in range(0, len(all_texts), batch_size)]

        results = []
        for chunk in tqdm(chunks, desc="Classifying"):
            batch_predictions = await classify_batch(client, url, chunk)
            if not batch_predictions:
                logger.error("Failed to get predictions for a batch. Using fallback.")
                first_label = settings.model.label_map[0]
                for _ in chunk:
                    results.append({"label": first_label, "confidence": 0.0})
            else:
                results.extend(batch_predictions)

    # 3. Process Results
    for i, sample in enumerate(samples):
        if i >= len(results):
            break

        pred = results[i]
        true_label = sample["true_label"]

        pred_label_str = pred["label"]
        pred_label = name_to_id.get(pred_label_str, 0)

        y_true.append(true_label)
        y_pred.append(pred_label)

    # 4. Calculate Metrics
    target_names = [settings.model.label_map[i] for i in range(settings.model.num_labels)]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Samples: {len(y_true)}")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Precision:     {prec:.4f}")
    print(f"Recall:        {rec:.4f}")
    print(f"F1-Score:      {f1:.4f}")
    print("-" * 60)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("-" * 60)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Accuracy/Quality")
    parser.add_argument("--url", default="http://localhost:8080", help="Inference API URL")
    parser.add_argument(
        "--test-dir", default="src/data/Test", help="Path to test data directory"
    )

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    test_dir = Path(args.test_dir)
    if not test_dir.is_absolute():
        test_dir = base_dir / test_dir

    if not test_dir.exists():
        logger.error(f"Directory not found: {test_dir}")
        return

    asyncio.run(run_benchmark(test_dir, args.url))


if __name__ == "__main__":
    main()

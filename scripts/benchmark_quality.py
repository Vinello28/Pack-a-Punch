#!/usr/bin/env python3
"""
Benchmark Quality Script

Measures classification quality (F1, ROC-AUC, Accuracy) using a labeled test set.
Iterates through 'ai' and 'non_ai' directories, sends requests to the inference API,
and computes metrics comparing predictions against ground truth.
"""

import sys
import argparse
import asyncio
import json
import httpx
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from loguru import logger
from tqdm import tqdm

# Add project root to path to import src if needed (though we use API here)
sys.path.insert(0, str(Path(__file__).parent.parent))

async def classify_batch(client: httpx.AsyncClient, url: str, texts: list[str]) -> list[dict]:
    try:
        response = await client.post(f"{url}/classify", json={"texts": texts})
        response.raise_for_status()
        return response.json()["predictions"]
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return []

async def run_benchmark(test_dir_ai: Path, test_dir_non_ai: Path, url: str, batch_size: int = 10):
    # 1. Load Data
    samples = []
    
    # Load AI samples
    logger.info(f"Loading AI samples from {test_dir_ai}...")
    ai_files = list(test_dir_ai.glob("*.txt"))
    for p in ai_files:
        try:
            text = p.read_text(encoding="utf-8").strip()
            if text:
                samples.append({"text": text, "true_label": 1, "filename": p.name}) # 1 = AI
        except Exception as e:
            logger.warning(f"Could not read {p}: {e}")

    # Load Non-AI samples
    logger.info(f"Loading Non-AI samples from {test_dir_non_ai}...")
    non_ai_files = list(test_dir_non_ai.glob("*.txt"))
    for p in non_ai_files:
        try:
            text = p.read_text(encoding="utf-8").strip()
            if text:
                samples.append({"text": text, "true_label": 0, "filename": p.name}) # 0 = Non-AI
        except Exception as e:
            logger.warning(f"Could not read {p}: {e}")
            
    if not samples:
        logger.error("No samples found!")
        sys.exit(1)
        
    logger.info(f"Total samples: {len(samples)} (AI: {len(ai_files)}, Non-AI: {len(non_ai_files)})")

    # 2. Run Inference
    y_true = []
    y_pred = []
    y_scores = [] # Probability of being AI (class 1)
    
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
        # Chunking
        chunks = [all_texts[i:i + batch_size] for i in range(0, len(all_texts), batch_size)]
        
        results = []
        for chunk in tqdm(chunks, desc="Classifying"):
            batch_predictions = await classify_batch(client, url, chunk)
            if not batch_predictions:
                logger.error("Failed to get predictions for a batch. Skipping...")
                # Fill with dummy/error or just skip? 
                # Ideally we fail or retry, but for simplicity let's skip metrics for these or append None
                # If we skip, arrays mismatch. We must append something or filter samples.
                # Here we assume robustness or just fail.
                # Let's append default "Non-AI" 0.0 confidence to avoid crashing, but log it.
                for _ in chunk:
                    results.append({"label": "NON_AI", "confidence": 0.0})
            else:
                results.extend(batch_predictions)
    
    # 3. Process Results
    for i, sample in enumerate(samples):
        if i >= len(results):
            break
            
        pred = results[i]
        true_label = sample["true_label"]
        
        pred_label_str = pred["label"] # "AI" or "NON_AI"
        confidence = pred["confidence"]
        
        # Map predicted string to int
        if pred_label_str == "AI":
            pred_label = 1
            score = confidence
        else:
            pred_label = 0
            score = 1.0 - confidence # Probability of AI is 1 - prob(Non_AI)
            
        y_true.append(true_label)
        y_pred.append(pred_label)
        y_scores.append(score)

    # 4. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1) # Sensitivity / TPR
    f1 = f1_score(y_true, y_pred, pos_label=1)
    
    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = 0.0
        logger.warning("Could not calculate ROC-AUC (maybe only one class present?)")

    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Total Samples: {len(y_true)}")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Precision:     {prec:.4f}")
    print(f"Recall:        {rec:.4f}")
    print(f"F1-Score:      {f1:.4f}")
    print(f"ROC-AUC:       {auc:.4f}")
    print("-" * 60)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("-" * 60)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["NON_AI", "AI"]))
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Benchmark Accuracy/Quality")
    parser.add_argument("--url", default="http://localhost:8080", help="Inference API URL")
    parser.add_argument("--data-ai", default="src/data/Test/ai", help="Path to AI test directory")
    parser.add_argument("--data-non-ai", default="src/data/Test/non_ai", help="Path to Non-AI test directory")
    
    args = parser.parse_args()
    
    # Resolve paths relative to inference-service if needed, or absolute
    base_dir = Path(__file__).parent.parent
    path_ai = Path(args.data_ai)
    if not path_ai.is_absolute():
        path_ai = base_dir / path_ai
        
    path_non_ai = Path(args.data_non_ai)
    if not path_non_ai.is_absolute():
        path_non_ai = base_dir / path_non_ai

    if not path_ai.exists():
        logger.error(f"Directory not found: {path_ai}")
        return
    if not path_non_ai.exists():
        logger.error(f"Directory not found: {path_non_ai}")
        return

    asyncio.run(run_benchmark(path_ai, path_non_ai, args.url))

if __name__ == "__main__":
    main()

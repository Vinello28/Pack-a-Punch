#!/usr/bin/env python3
"""
Benchmark script for Pack-a-Punch classifier using real test data.

Measures accuracy, precision, recall, F1-score, as well as throughput and latency.

Usage:
    python scripts/benchmark_test_set.py --url http://localhost:8080
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import asyncio
from statistics import mean, stdev
from collections import Counter

import httpx
from loguru import logger
from tqdm import tqdm

from src.config import settings

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Pack-a-Punch on real test data")
    
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080",
        help="API URL",
    )
    
    parser.add_argument(
        "--test-data-dir",
        type=str,
        default="src/data/test",
        help="Path to test data directory",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.inference.batch_size,
        help="Batch size for inference",
    )
    
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=10,
        help="Number of concurrent requests",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and print stats without calling API",
    )
    
    return parser.parse_args()


def load_test_data(data_dir: Path) -> list[tuple[str, str]]:
    """
    Load test data from ai/ and non_ai/ directories.
    Returns: list of (text, label)
    """
    data = []
    
    ai_dir = data_dir / "ai"
    non_ai_dir = data_dir / "non_ai"
    
    if ai_dir.exists():
        for file_path in ai_dir.rglob("*.txt"):
            try:
                text = file_path.read_text(encoding="utf-8").strip()
                if text:
                    data.append((text, "AI"))
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
                
    if non_ai_dir.exists():
        for file_path in non_ai_dir.rglob("*.txt"):
            try:
                text = file_path.read_text(encoding="utf-8").strip()
                if text:
                    data.append((text, "NON_AI"))
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
                
    return data


async def run_benchmark(
    url: str,
    test_data: list[tuple[str, str]],
    batch_size: int,
    concurrency: int,
) -> tuple[float, list[float], list[tuple[str, str]], int]:
    """
    Run benchmark against API.
    Returns: (total_time, latencies, results, error_count)
    results is a list of (predicted_label, ground_truth_label)
    """
    latencies = []
    all_predictions = []
    error_count = 0
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Create batches of (text, ground_truth)
        batches = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]
        
        total_start = time.perf_counter()
        
        # Process in chunks of concurrent requests
        for i in tqdm(range(0, len(batches), concurrency), desc="Evaluating Test Set"):
            chunk = batches[i:i + concurrency]
            
            tasks = []
            for batch in chunk:
                texts = [item[0] for item in batch]
                tasks.append(client.post(f"{url}/classify", json={"texts": texts}))
            
            chunk_start = time.perf_counter()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            chunk_end = time.perf_counter()
            
            for idx, resp in enumerate(responses):
                if isinstance(resp, Exception):
                    logger.error(f"Request failed: {resp}")
                    error_count += 1
                elif resp.status_code != 200:
                    logger.error(f"API error: {resp.status_code} - {resp.text}")
                    error_count += 1
                else:
                    # Successful request
                    data = resp.json()
                    preds = data["predictions"]
                    
                    # Store latencies (total request time / batch_size)
                    latencies.append((chunk_end - chunk_start) * 1000 / len(chunk))
                    
                    # Store (prediction, ground_truth) for metrics
                    batch_ground_truth = [item[1] for item in chunk[idx]]
                    for p, gt in zip(preds, batch_ground_truth):
                        all_predictions.append((p["label"], gt))

    total_time = time.perf_counter() - total_start
    return total_time, latencies, all_predictions, error_count


def calculate_metrics(results: list[tuple[str, str]]):
    """Calculate accuracy, precision, recall, F1."""
    if not results:
        return {}
        
    tp = sum(1 for p, gt in results if p == "AI" and gt == "AI")
    tn = sum(1 for p, gt in results if p == "NON_AI" and gt == "NON_AI")
    fp = sum(1 for p, gt in results if p == "AI" and gt == "NON_AI")
    fn = sum(1 for p, gt in results if p == "NON_AI" and gt == "AI")
    
    accuracy = (tp + tn) / len(results)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "counts": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    }


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Pack-a-Punch Evaluation (Test Set)")
    logger.info("=" * 60)
    logger.info(f"Target URL: {args.url}")
    
    # Load test data
    test_data_path = Path(args.test_data_dir)
    logger.info(f"Loading test data from {test_data_path}...")
    test_data = load_test_data(test_data_path)
    
    if not test_data:
        logger.error("No test data found!")
        return 1
        
    logger.info(f"Loaded {len(test_data)} test samples.")
    ai_count = sum(1 for _, label in test_data if label == "AI")
    non_ai_count = len(test_data) - ai_count
    logger.info(f"AI: {ai_count}, NON_AI: {non_ai_count}")
    
    if args.dry_run:
        logger.info("Dry run complete. Exiting.")
        return 0
        
    # Run benchmark
    logger.info(f"Running evaluation with batch_size={args.batch_size}...")
    
    try:
        total_time, latencies, results, error_count = asyncio.run(run_benchmark(
            args.url,
            test_data,
            args.batch_size,
            args.concurrent_requests
        ))
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    success_count = len(results)
    throughput = success_count / total_time if total_time > 0 else 0
    
    avg_latency = mean(latencies) if latencies else 0
    std_latency = stdev(latencies) if len(latencies) > 1 else 0
    
    # Print results
    logger.info("=" * 60)
    logger.info("ACCURACY METRICS")
    logger.info("=" * 60)
    if metrics:
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-score:  {metrics['f1']:.4f}")
        logger.info("-" * 60)
        logger.info(f"Confusion Matrix:")
        logger.info(f"  TP: {metrics['counts']['tp']}  FP: {metrics['counts']['fp']}")
        logger.info(f"  FN: {metrics['counts']['fn']}  TN: {metrics['counts']['tn']}")
    else:
        logger.warning("No metrics calculated due to missing results.")
        
    logger.info("=" * 60)
    logger.info("PERFORMANCE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total samples:     {len(test_data):,}")
    logger.info(f"Successful:        {success_count:,}")
    logger.info(f"Errors (batches):  {error_count}")
    logger.info(f"Total time:        {total_time:.2f}s")
    logger.info(f"Throughput:        {throughput:,.2f} samples/sec")
    logger.info(f"Avg latency:       {avg_latency:.2f}ms ± {std_latency:.2f}ms")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

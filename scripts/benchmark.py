#!/usr/bin/env python3
"""
Benchmark script for Pack-a-Punch classifier.

Measures throughput and latency for inference.

Usage:
    python scripts/benchmark.py --num-samples 10000 --batch-size 64
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import asyncio
from statistics import mean, stdev

import httpx
from loguru import logger
from tqdm import tqdm

from src.config import settings
# from src.inference.engine import create_engine


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Pack-a-Punch inference")
    
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080",
        help="API URL",
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples to process",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.inference.batch_size,
        help="Batch size for inference",
    )
    
    parser.add_argument(
        "--num-sessions",
        type=int,
        default=settings.inference.num_sessions,
        help="Number of parallel sessions (for simulation)",
    )
    
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=10,
        help="Number of concurrent requests",
    )
    
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    
    parser.add_argument(
        "--text-length",
        type=int,
        default=200,
        help="Approximate text length in words",
    )
    
    return parser.parse_args()


def generate_dummy_texts(num_samples: int, text_length: int) -> list[str]:
    """Generate dummy texts for benchmarking."""
    base_text = (
        "Questo è un testo di esempio per il benchmark del classificatore. "
        "Contiene diverse frasi che simulano un documento reale da classificare. "
        "Il modello deve analizzare questo contenuto e determinare se è stato "
        "generato da un'intelligenza artificiale o scritto da un umano. "
    )
    
    # Repeat to reach desired length
    words_per_base = len(base_text.split())
    repeats = max(1, text_length // words_per_base)
    full_text = base_text * repeats
    
    return [full_text] * num_samples


async def run_benchmark_api(
    url: str,
    texts: list[str],
    batch_size: int,
    concurrency: int,
) -> tuple[float, list[float], int]:
    """
    Run benchmark against API.
    Returns: (total_time, latencies, error_count)
    """
    latencies = []
    error_count = 0
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Create batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        total_start = time.perf_counter()
        
        # Process in chunks of concurrent requests
        for i in tqdm(range(0, len(batches), concurrency), desc="Benchmarking API"):
            chunk = batches[i:i + concurrency]
            
            tasks = []
            for batch in chunk:
                tasks.append(client.post(f"{url}/classify", json={"texts": batch}))
            
            chunk_start = time.perf_counter()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            chunk_end = time.perf_counter()
            
            for resp in responses:
                if isinstance(resp, Exception):
                    error_count += 1
                elif resp.status_code != 200:
                    error_count += 1
                else:
                    # Only record latency for successful requests
                    # Using chunk average for simplicity
                    latencies.append((chunk_end - chunk_start) * 1000 / len(chunk))

    total_time = time.perf_counter() - total_start
    return total_time, latencies, error_count


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Pack-a-Punch Benchmark (API Mode)")
    logger.info("=" * 60)
    logger.info(f"Target URL: {args.url}")
    
    # Generate test data
    logger.info(f"Generating {args.num_samples} test samples...")
    texts = generate_dummy_texts(args.num_samples, args.text_length)
    
    # Run benchmark
    logger.info(f"Running benchmark with {args.concurrent_requests} concurrent requests...")
    
    try:
        total_time, latencies, error_count = asyncio.run(run_benchmark_api(
            args.url,
            texts,
            args.batch_size,
            args.concurrent_requests
        ))
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1
    
    # Calculate metrics
    success_count = args.num_samples - (error_count * args.batch_size) # Approximate samples lost
    throughput = success_count / total_time if total_time > 0 else 0
    
    avg_latency = mean(latencies) if latencies else 0
    std_latency = stdev(latencies) if len(latencies) > 1 else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    
    # Print results
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total samples:     {args.num_samples:,}")
    logger.info(f"Successful:        {success_count:,}")
    logger.info(f"Errors (batches):  {error_count} (approx {error_count * args.batch_size} samples)")
    logger.info(f"Batch size:        {args.batch_size}")
    logger.info(f"Concurrency:       {args.concurrent_requests}")
    logger.info(f"Total time:        {total_time:.2f}s")
    logger.info(f"Throughput:        {throughput:,.0f} samples/sec")
    logger.info("-" * 60)
    logger.info(f"Avg latency:       {avg_latency:.2f}ms ± {std_latency:.2f}ms")
    logger.info(f"Min latency:       {min_latency:.2f}ms")
    logger.info(f"Max latency:       {max_latency:.2f}ms")
    logger.info("=" * 60)
    
    # Estimate for 20M records
    time_20m = 20_000_000 / throughput if throughput > 0 else 0
    logger.info(f"Estimated time for 20M records: {time_20m / 60:.1f} minutes")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

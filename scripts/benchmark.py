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

from loguru import logger
from tqdm import tqdm

from src.config import settings
from src.inference.engine import create_engine


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Pack-a-Punch inference")
    
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
        help="Number of parallel sessions",
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


async def run_benchmark_async(
    engine,
    texts: list[str],
    batch_size: int,
) -> tuple[float, list[float]]:
    """Run async benchmark."""
    latencies = []
    
    total_start = time.perf_counter()
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Benchmarking"):
        batch = texts[i:i + batch_size]
        
        batch_start = time.perf_counter()
        await engine.predict_parallel(batch)
        batch_end = time.perf_counter()
        
        latencies.append((batch_end - batch_start) * 1000)
    
    total_time = time.perf_counter() - total_start
    
    return total_time, latencies


def run_benchmark_sync(
    engine,
    texts: list[str],
    batch_size: int,
) -> tuple[float, list[float]]:
    """Run synchronous benchmark."""
    latencies = []
    
    total_start = time.perf_counter()
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Benchmarking"):
        batch = texts[i:i + batch_size]
        
        batch_start = time.perf_counter()
        engine.predict_batch(batch)
        batch_end = time.perf_counter()
        
        latencies.append((batch_end - batch_start) * 1000)
    
    total_time = time.perf_counter() - total_start
    
    return total_time, latencies


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Pack-a-Punch Benchmark")
    logger.info("=" * 60)
    
    # Create engine
    engine = create_engine(
        num_sessions=args.num_sessions,
        batch_size=args.batch_size,
    )
    
    # Generate test data
    logger.info(f"Generating {args.num_samples} test samples...")
    texts = generate_dummy_texts(args.num_samples, args.text_length)
    
    # Warmup
    logger.info(f"Warming up ({args.warmup} iterations)...")
    engine.warmup(args.warmup)
    
    # Run benchmark
    logger.info("Running benchmark...")
    total_time, latencies = run_benchmark_sync(
        engine,
        texts,
        args.batch_size,
    )
    
    # Calculate metrics
    throughput = args.num_samples / total_time
    avg_latency = mean(latencies)
    std_latency = stdev(latencies) if len(latencies) > 1 else 0
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    # Print results
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total samples:     {args.num_samples:,}")
    logger.info(f"Batch size:        {args.batch_size}")
    logger.info(f"Parallel sessions: {args.num_sessions}")
    logger.info(f"Total time:        {total_time:.2f}s")
    logger.info(f"Throughput:        {throughput:,.0f} samples/sec")
    logger.info("-" * 60)
    logger.info(f"Avg batch latency: {avg_latency:.2f}ms ± {std_latency:.2f}ms")
    logger.info(f"Min batch latency: {min_latency:.2f}ms")
    logger.info(f"Max batch latency: {max_latency:.2f}ms")
    logger.info("=" * 60)
    
    # Estimate for 20M records
    time_20m = 20_000_000 / throughput
    logger.info(f"Estimated time for 20M records: {time_20m / 60:.1f} minutes")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

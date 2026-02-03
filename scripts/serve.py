#!/usr/bin/env python3
"""
Inference server script for Pack-a-Punch classifier.

Usage:
    python scripts/serve.py
    python scripts/serve.py --port 8080 --num-sessions 2
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import uvicorn
from loguru import logger

from src.config import settings


def parse_args():
    parser = argparse.ArgumentParser(description="Run Pack-a-Punch inference server")
    
    parser.add_argument(
        "--host",
        type=str,
        default=settings.server.host,
        help="Server host",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=settings.server.port,
        help="Server port",
    )
    
    parser.add_argument(
        "--num-sessions",
        type=int,
        default=settings.inference.num_sessions,
        help="Number of parallel ONNX sessions",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.inference.batch_size,
        help="Inference batch size",
    )
    
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to ONNX model",
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Update settings
    settings.inference.num_sessions = args.num_sessions
    settings.inference.batch_size = args.batch_size
    
    logger.info("=" * 60)
    logger.info("Pack-a-Punch Inference Server")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Sessions: {args.num_sessions}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 60)
    
    uvicorn.run(
        "src.inference.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,  # Single worker for GPU
        log_level="info",
    )


if __name__ == "__main__":
    main()

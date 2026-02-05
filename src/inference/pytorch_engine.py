"""
PyTorch-based inference engine for comparison with ONNX Runtime.

This engine uses PyTorch directly with CUDA for inference,
useful for performance comparison and debugging ONNX issues.
"""

from pathlib import Path
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from loguru import logger

from src.config import settings


class PyTorchInferenceEngine:
    """
    PyTorch-based inference engine using CUDA.
    
    This is an alternative to the ONNX-based engine for
    performance comparison and debugging.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        batch_size: int = 64,
        device: str = "cuda",
        fp16: bool = True,
    ):
        self.batch_size = batch_size
        self.device = device
        self.fp16 = fp16
        
        # Find model path
        if model_path is None:
            model_path = self._find_model()
        
        self.model_path = Path(model_path)
        
        logger.info(f"Loading PyTorch model from {self.model_path}")
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if fp16 else torch.float32,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Move to device
        self.model = self.model.to(device)
        self.model.eval()
        
        # torch.compile disabled - causes crashes in container environment
        # Eager mode is sufficient for benchmarking PyTorch vs ONNX
        
        # Thread pool for async inference
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info(
            f"PyTorchInferenceEngine initialized: "
            f"device={device}, fp16={fp16}, batch_size={batch_size}"
        )
    
    def _find_model(self) -> Path:
        """Find PyTorch model directory."""
        models_dir = settings.models_dir
        
        # Check for model files
        if (models_dir / "config.json").exists():
            logger.info(f"Found model: {models_dir}")
            return models_dir
        
        # Check subdirectories
        for subdir in models_dir.iterdir():
            if subdir.is_dir() and (subdir / "config.json").exists():
                logger.info(f"Found model: {subdir}")
                return subdir
        
        raise FileNotFoundError(
            f"No PyTorch model found in {models_dir}. "
            "Look for config.json and model.safetensors files."
        )
    
    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize texts for inference."""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=settings.model.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Move to device
        return {
            k: v.to(self.device) for k, v in encodings.items()
        }
    
    @torch.inference_mode()
    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Synchronous batch prediction.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of prediction dicts with label and confidence
        """
        if not texts:
            return []
        
        # Tokenize
        inputs = self._tokenize(texts)
        
        # Run inference
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Apply softmax
        probs = torch.softmax(logits, dim=-1)
        
        # Get predictions
        pred_ids = torch.argmax(probs, dim=-1)
        confidences = torch.max(probs, dim=-1).values
        
        # Move to CPU and convert
        pred_ids = pred_ids.cpu().numpy()
        confidences = confidences.cpu().numpy()
        
        results = []
        for pred_id, confidence in zip(pred_ids, confidences):
            label = settings.model.label_map[int(pred_id)]
            results.append({
                "label": label,
                "confidence": float(confidence),
            })
        
        return results
    
    async def predict_batch_async(self, texts: list[str]) -> list[dict]:
        """
        Asynchronous batch prediction.
        
        Runs inference in thread pool to avoid blocking event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.predict_batch,
            texts,
        )
    
    async def predict_parallel(self, texts: list[str]) -> list[dict]:
        """
        Parallel prediction for large batches.
        
        Splits input into chunks for batch processing.
        """
        if len(texts) <= self.batch_size:
            return await self.predict_batch_async(texts)
        
        # Split into chunks
        chunks = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        
        # Process sequentially (GPU can't parallelize well with PyTorch)
        results = []
        for chunk in chunks:
            chunk_results = await self.predict_batch_async(chunk)
            results.extend(chunk_results)
        
        return results
    
    def warmup(self, num_iterations: int = 3):
        """Warmup with dummy data."""
        logger.info("Warming up PyTorch inference engine...")
        dummy_texts = ["Testo di prova per warmup"] * min(self.batch_size, 8)
        
        for i in range(num_iterations):
            self.predict_batch(dummy_texts)
        
        # Sync CUDA
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        logger.info("Warmup complete")
    
    def get_stats(self) -> dict:
        """Get engine statistics."""
        stats = {
            "model_path": str(self.model_path),
            "backend": "pytorch",
            "batch_size": self.batch_size,
            "device": self.device,
            "fp16": self.fp16,
        }
        
        if self.device == "cuda":
            stats["cuda_device"] = torch.cuda.get_device_name(0)
            stats["cuda_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            stats["cuda_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        
        return stats


def create_pytorch_engine(**kwargs) -> PyTorchInferenceEngine:
    """Factory function to create PyTorch inference engine."""
    return PyTorchInferenceEngine(
        batch_size=kwargs.get("batch_size", settings.inference.batch_size),
        device=kwargs.get("device", settings.inference.device),
        fp16=kwargs.get("fp16", settings.inference.use_fp16),
        model_path=kwargs.get("model_path"),
    )

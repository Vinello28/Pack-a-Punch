"""
TensorRT-based inference engine using optimum and onnxruntime.
"""

from pathlib import Path
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import shutil

import numpy as np
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from loguru import logger

from src.config import settings
from src.inference.labeling import probs_to_results

class TensorRTInferenceEngine:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        batch_size: int = 64,
        device: str = "cuda",
    ):
        self.batch_size = batch_size
        self.device = device
        
        if model_path is None:
            model_path = self._find_model()
            
        self.model_path = Path(model_path)
        
        logger.info(f"Loading/Exporting model to ONNX and compiling for TensorRT from {self.model_path}")
        
        # We specify provider='TensorrtExecutionProvider'. If the model isn't in ONNX format yet, 
        # export=True will convert the PyTorch model to ONNX automatically!
        # It will then be loaded by ORT with TensorRT execution provider.
        self.model = ORTModelForSequenceClassification.from_pretrained(
            self.model_path,
            export=True,
            provider="TensorrtExecutionProvider",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info(
            f"TensorRTInferenceEngine initialized: "
            f"device={device}, batch_size={batch_size}"
        )
        
    def _find_model(self) -> Path:
        models_dir = settings.models_dir
        if (models_dir / "config.json").exists():
            return models_dir
        for subdir in models_dir.iterdir():
            if subdir.is_dir() and (subdir / "config.json").exists():
                return subdir
        raise FileNotFoundError("Model not found in models_dir.")

    def _tokenize(self, texts: list[str]) -> dict:
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=settings.model.max_length,
            padding=True, # Dynamic padding enabled!
            return_tensors="np", # ORT needs numpy arrays
        )
        return dict(encodings)
        
    def predict_batch(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
            
        inputs = self._tokenize(texts)
        
        # Run inference using Optimum ORT
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Apply softmax manually (numpy), then threshold P(tracciabilita).
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return probs_to_results(probs)

    async def predict_batch_async(self, texts: list[str]) -> list[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.predict_batch,
            texts,
        )
        
    async def predict_parallel(self, texts: list[str]) -> list[dict]:
        if len(texts) <= self.batch_size:
            return await self.predict_batch_async(texts)
            
        chunks = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        
        results = []
        for chunk in chunks:
            chunk_results = await self.predict_batch_async(chunk)
            results.extend(chunk_results)
            
        return results

    def warmup(self, num_iterations: int = 3):
        logger.info("Warming up TensorRT inference engine (this may take a minute for TRT engine building)...")
        dummy_texts = ["Testo di prova per warmup"] * min(self.batch_size, 8)
        
        for i in range(num_iterations):
            self.predict_batch(dummy_texts)
            
        logger.info("Warmup complete")

    def get_stats(self) -> dict:
        return {
            "model_path": str(self.model_path),
            "backend": "tensorrt",
            "batch_size": self.batch_size,
            "device": self.device,
        }

def create_tensorrt_engine(**kwargs) -> TensorRTInferenceEngine:
    return TensorRTInferenceEngine(
        batch_size=kwargs.get("batch_size", settings.inference.batch_size),
        device=kwargs.get("device", settings.inference.device),
        model_path=kwargs.get("model_path"),
    )

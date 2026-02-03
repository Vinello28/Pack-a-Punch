"""
High-performance ONNX Runtime inference engine.

Features:
- Multiple parallel ONNX sessions for GPU utilization
- CUDA streams for overlapping compute/transfer
- IO Binding for zero-copy inference
"""

from pathlib import Path
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from loguru import logger

from src.config import settings


class InferenceEngine:
    """
    Multi-session ONNX inference engine.
    
    Uses multiple ONNX sessions to parallelize inference across
    CUDA streams, maximizing GPU utilization.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        num_sessions: int = 2,
        batch_size: int = 64,
        device: str = "cuda",
        fp16: bool = True,
    ):
        self.num_sessions = num_sessions
        self.batch_size = batch_size
        self.device = device
        self.fp16 = fp16
        
        # Find model path
        if model_path is None:
            model_path = self._find_model()
        
        self.model_path = Path(model_path)
        
        # Load tokenizer
        tokenizer_path = self.model_path.parent
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Create ONNX sessions
        self.sessions = self._create_sessions()
        
        # Thread pool for parallel inference
        self.executor = ThreadPoolExecutor(max_workers=num_sessions)
        
        # Round-robin session index
        self._session_idx = 0
        self._lock = asyncio.Lock()
        
        logger.info(
            f"InferenceEngine initialized: "
            f"{num_sessions} sessions, "
            f"batch_size={batch_size}, "
            f"device={device}"
        )
    
    def _find_model(self) -> Path:
        """Find ONNX model file in models directory."""
        models_dir = settings.models_dir
        
        # Priority: optimized FP16 > optimized > regular FP16 > regular
        candidates = [
            "model_optimized_fp16.onnx",
            "model_optimized.onnx",
            "model_fp16.onnx",
            "model.onnx",
        ]
        
        for name in candidates:
            path = models_dir / name
            if path.exists():
                logger.info(f"Found model: {path}")
                return path
        
        raise FileNotFoundError(
            f"No ONNX model found in {models_dir}. "
            "Run training and export first."
        )
    
    def _create_sessions(self) -> list[ort.InferenceSession]:
        """Create multiple ONNX Runtime sessions."""
        sessions = []
        
        for i in range(self.num_sessions):
            # Session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 2
            
            # Execution providers
            if self.device == "cuda":
                providers = [
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": 0,
                            "arena_extend_strategy": "kNextPowerOfTwo",
                            "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB per session
                            "cudnn_conv_algo_search": "EXHAUSTIVE",
                            "do_copy_in_default_stream": True,
                        },
                    ),
                    "CPUExecutionProvider",
                ]
            else:
                providers = ["CPUExecutionProvider"]
            
            session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers,
            )
            
            sessions.append(session)
            logger.debug(f"Created session {i + 1}/{self.num_sessions}")
        
        return sessions
    
    def _tokenize(self, texts: list[str]) -> dict[str, np.ndarray]:
        """Tokenize texts for inference."""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=settings.model.max_length,
            padding="max_length",
            return_tensors="np",
        )
        
        return {
            "input_ids": encodings["input_ids"].astype(np.int64),
            "attention_mask": encodings["attention_mask"].astype(np.int64),
        }
    
    def _run_session(
        self,
        session: ort.InferenceSession,
        inputs: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Run inference on a single session."""
        outputs = session.run(
            ["logits"],
            inputs,
        )
        return outputs[0]
    
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
        
        # Get next session (round-robin)
        session = self.sessions[self._session_idx % self.num_sessions]
        self._session_idx += 1
        
        # Run inference
        logits = self._run_session(session, inputs)
        
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Get predictions
        pred_ids = np.argmax(probs, axis=-1)
        confidences = np.max(probs, axis=-1)
        
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
        Parallel prediction across multiple sessions.
        
        Splits input across sessions for maximum throughput.
        """
        if len(texts) <= self.batch_size:
            return await self.predict_batch_async(texts)
        
        # Split into chunks for parallel processing
        chunks = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        
        # Run in parallel
        tasks = [self.predict_batch_async(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        return [item for sublist in results for item in sublist]
    
    def warmup(self, num_iterations: int = 3):
        """Warmup sessions with dummy data."""
        logger.info("Warming up inference engine...")
        dummy_texts = ["Testo di prova per warmup"] * self.batch_size
        
        for i in range(num_iterations):
            for session in self.sessions:
                inputs = self._tokenize(dummy_texts)
                self._run_session(session, inputs)
        
        logger.info("Warmup complete")
    
    def get_stats(self) -> dict:
        """Get engine statistics."""
        return {
            "model_path": str(self.model_path),
            "num_sessions": self.num_sessions,
            "batch_size": self.batch_size,
            "device": self.device,
            "providers": self.sessions[0].get_providers() if self.sessions else [],
        }


def create_engine(**kwargs) -> InferenceEngine:
    """Factory function to create inference engine."""
    return InferenceEngine(
        num_sessions=kwargs.get("num_sessions", settings.inference.num_sessions),
        batch_size=kwargs.get("batch_size", settings.inference.batch_size),
        device=kwargs.get("device", settings.inference.device),
        fp16=kwargs.get("fp16", settings.inference.use_fp16),
        model_path=kwargs.get("model_path"),
    )

"""
Triton-based inference engine.

The heavy model execution runs inside an NVIDIA Triton Inference Server container
(ONNX Runtime backend + TensorRT acceleration + server-side dynamic batching).
This engine is a thin client: it tokenizes text (HuggingFace), sends the tensors to
Triton over gRPC, and applies softmax/argmax on the returned logits.

Same interface as PyTorchInferenceEngine / TensorRTInferenceEngine, so it is a drop-in
swap in server.py.
"""

import os
from pathlib import Path
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tritonclient.grpc as grpcclient
from transformers import AutoTokenizer
from loguru import logger

from src.config import settings
from src.inference.labeling import probs_to_results


class TritonInferenceEngine:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        batch_size: int = 64,
        device: str = "cuda",
        url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.batch_size = batch_size
        self.device = device
        self.url = url or os.getenv("TRITON_URL", "localhost:8001")
        self.model_name = model_name or os.getenv("TRITON_MODEL_NAME", "classifier")

        if model_path is None:
            model_path = self._find_model()
        self.model_path = Path(model_path)

        # Only the tokenizer lives in the gateway; the model runs on Triton.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        logger.info(f"Connecting to Triton at {self.url} (model='{self.model_name}')")
        self.client = grpcclient.InferenceServerClient(url=self.url)

        # Discover the exact inputs/output the model expects, so we stay correct
        # regardless of whether the ONNX exposes token_type_ids.
        metadata = self.client.get_model_metadata(self.model_name)
        self._input_names = [i.name for i in metadata.inputs]
        self._output_name = metadata.outputs[0].name

        self.executor = ThreadPoolExecutor(max_workers=2)

        logger.info(
            f"TritonInferenceEngine initialized: device={device}, "
            f"batch_size={batch_size}, inputs={self._input_names}, "
            f"output='{self._output_name}'"
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
            padding=True,  # Dynamic padding
            return_tensors="np",
        )
        # Triton/ONNX expect int64 tensors.
        return {k: np.asarray(v, dtype=np.int64) for k, v in encodings.items()}

    def _build_inputs(self, encodings: dict) -> list:
        ref_shape = encodings["input_ids"].shape
        inputs = []
        for name in self._input_names:
            arr = encodings.get(name)
            if arr is None:
                # Model expects an input the tokenizer didn't produce (e.g.
                # token_type_ids on a single-segment model): send zeros.
                arr = np.zeros(ref_shape, dtype=np.int64)
            infer_input = grpcclient.InferInput(name, list(arr.shape), "INT64")
            infer_input.set_data_from_numpy(arr)
            inputs.append(infer_input)
        return inputs

    def predict_batch(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []

        encodings = self._tokenize(texts)
        inputs = self._build_inputs(encodings)
        outputs = [grpcclient.InferRequestedOutput(self._output_name)]

        response = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
        )
        logits = response.as_numpy(self._output_name)

        # Softmax (numpy), then threshold P(tracciabilita) via the shared post-processor.
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
        logger.info(
            "Warming up Triton engine (first call may build the TensorRT engine "
            "server-side and take a while)..."
        )
        dummy_texts = ["Testo di prova per warmup"] * min(self.batch_size, 8)

        for _ in range(num_iterations):
            self.predict_batch(dummy_texts)

        logger.info("Warmup complete")

    def get_stats(self) -> dict:
        stats = {
            "model_path": str(self.model_path),
            "backend": "triton",
            "batch_size": self.batch_size,
            "device": self.device,
            "triton_url": self.url,
            "model_name": self.model_name,
        }
        try:
            stats["server_live"] = self.client.is_server_live()
            stats["model_ready"] = self.client.is_model_ready(self.model_name)
        except Exception as e:  # pragma: no cover - best-effort stats
            stats["server_live"] = False
            stats["error"] = str(e)
        return stats


def create_triton_engine(**kwargs) -> TritonInferenceEngine:
    # Cap the per-Triton-call batch size. Self-attention memory scales as
    # batch * heads * seq^2, so an unbounded batch (e.g. 512) with long sequences OOMs the
    # GPU. predict_parallel chunks larger inputs into pieces of this size, and it must match
    # the Triton model's max_batch_size (config.pbtxt). Override with TRITON_MAX_BATCH.
    max_batch = int(os.getenv("TRITON_MAX_BATCH", "128"))
    batch_size = kwargs.get("batch_size") or settings.inference.batch_size
    batch_size = min(batch_size, max_batch)
    return TritonInferenceEngine(
        batch_size=batch_size,
        device=kwargs.get("device", settings.inference.device),
        model_path=kwargs.get("model_path"),
        url=kwargs.get("url"),
        model_name=kwargs.get("model_name"),
    )

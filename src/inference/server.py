"""
FastAPI server for text classification inference.

Supports both ONNX Runtime and PyTorch backends.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from src.config import settings
from .engine import InferenceEngine, create_engine
from .pytorch_engine import PyTorchInferenceEngine, create_pytorch_engine
from .batching import DynamicBatcher

# Global state
_engine: Optional[Union[InferenceEngine, PyTorchInferenceEngine]] = None
_batcher: Optional[DynamicBatcher] = None

# Backend selection (set via environment variable)
BACKEND = os.environ.get("INFERENCE_BACKEND", "onnx").lower()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _engine, _batcher
    
    logger.info("Starting inference server...")
    logger.info(f"Backend: {BACKEND}")
    
    try:
        # Initialize engine based on backend selection
        if BACKEND == "pytorch":
            logger.info("Using PyTorch backend")
            _engine = create_pytorch_engine()
        else:
            logger.info("Using ONNX Runtime backend")
            _engine = create_engine()
        
        _engine.warmup()
        
        # Initialize batcher
        _batcher = DynamicBatcher(
            inference_fn=_engine.predict_parallel,
            max_batch_size=settings.inference.batch_size,
            timeout_ms=settings.inference.batch_timeout_ms,
            max_queue_size=settings.inference.max_queue_size,
        )
        
        logger.info("Server ready")
        yield
        
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise
    finally:
        logger.info("Shutting down...")


app = FastAPI(
    title="Pack-a-Punch Classifier",
    description="Binary text classification API (AI/NON_AI)",
    version="0.1.0",
    lifespan=lifespan,
)


# Request/Response models
class ClassifyRequest(BaseModel):
    """Classification request."""
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=512,  # Increased from 100 to allow larger batches
        description="List of texts to classify",
    )


class Prediction(BaseModel):
    """Single prediction result."""
    label: str = Field(..., description="Predicted label (AI or NON_AI)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class ClassifyResponse(BaseModel):
    """Classification response."""
    predictions: list[Prediction]
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    engine_stats: dict


class MetricsResponse(BaseModel):
    """Metrics response."""
    engine: dict
    batcher: dict


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest) -> ClassifyResponse:
    """
    Classify texts as AI-generated or human-written.
    
    Accepts up to 100 texts per request. Uses dynamic batching
    for optimal throughput.
    """
    if _batcher is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    start_time = time.perf_counter()
    
    try:
        results = await _batcher.submit(request.texts)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return ClassifyResponse(
            predictions=[Prediction(**r) for r in results],
            processing_time_ms=round(processing_time, 2),
        )
        
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if _engine is not None else "unhealthy",
        model_loaded=_engine is not None,
        engine_stats=_engine.get_stats() if _engine else {},
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Get server metrics."""
    return MetricsResponse(
        engine=_engine.get_stats() if _engine else {},
        batcher=_batcher.get_stats() if _batcher else {},
    )


# Simple endpoint for quick testing
@app.post("/classify/single")
async def classify_single(text: str) -> Prediction:
    """Classify a single text (convenience endpoint)."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    results = _engine.predict_batch([text])
    return Prediction(**results[0])

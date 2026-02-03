"""
Dynamic batching manager for optimal throughput.

Accumulates incoming requests into batches to maximize
GPU utilization while respecting latency constraints.
"""

import asyncio
from typing import Optional
from dataclasses import dataclass, field
from time import perf_counter

from loguru import logger

from src.config import settings


@dataclass
class BatchRequest:
    """Single request in the batch queue."""
    texts: list[str]
    future: asyncio.Future = field(default_factory=asyncio.Future)
    timestamp: float = field(default_factory=perf_counter)


class DynamicBatcher:
    """
    Collects requests and processes them in optimal batches.
    
    Waits for either:
    - max_batch_size requests accumulated
    - timeout_ms elapsed since first request
    """
    
    def __init__(
        self,
        inference_fn,
        max_batch_size: int = 64,
        timeout_ms: int = 50,
        max_queue_size: int = 1000,
    ):
        self.inference_fn = inference_fn
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.max_queue_size = max_queue_size
        
        self._queue: list[BatchRequest] = []
        self._lock = asyncio.Lock()
        self._processing = False
        self._batch_task: Optional[asyncio.Task] = None
        
        # Stats
        self._total_requests = 0
        self._total_batches = 0
        self._total_texts = 0
    
    async def submit(self, texts: list[str]) -> list[dict]:
        """
        Submit texts for classification.
        
        Returns predictions when batch is processed.
        """
        if len(self._queue) >= self.max_queue_size:
            raise RuntimeError("Batch queue full, try again later")
        
        request = BatchRequest(texts=texts)
        
        async with self._lock:
            self._queue.append(request)
            self._total_requests += 1
            self._total_texts += len(texts)
            
            # Start batch processor if not running
            if self._batch_task is None or self._batch_task.done():
                self._batch_task = asyncio.create_task(self._process_batches())
        
        return await request.future
    
    async def _process_batches(self):
        """Background task to process batched requests."""
        while True:
            await asyncio.sleep(self.timeout_ms / 1000)
            
            async with self._lock:
                if not self._queue:
                    return
                
                # Collect all pending requests
                requests = self._queue.copy()
                self._queue.clear()
            
            # Flatten all texts
            all_texts = []
            request_indices = []  # Maps text index to request
            
            for req_idx, request in enumerate(requests):
                for _ in request.texts:
                    request_indices.append(req_idx)
                all_texts.extend(request.texts)
            
            try:
                # Run inference on all texts
                all_results = await self.inference_fn(all_texts)
                
                # Distribute results back to requests
                for request in requests:
                    num_texts = len(request.texts)
                    request_results = all_results[:num_texts]
                    all_results = all_results[num_texts:]
                    request.future.set_result(request_results)
                
                self._total_batches += 1
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                for request in requests:
                    if not request.future.done():
                        request.future.set_exception(e)
    
    def get_stats(self) -> dict:
        """Get batcher statistics."""
        return {
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "total_texts": self._total_texts,
            "queue_size": len(self._queue),
            "avg_texts_per_batch": (
                self._total_texts / self._total_batches
                if self._total_batches > 0
                else 0
            ),
        }

"""
LLM Teacher Distillation Pipeline.

Uses a local LLM (via LM Studio or similar) to label unlabeled texts,
creating a high-quality training dataset for the student ModernBERT.
"""

import json
import asyncio
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger
from tqdm import tqdm

from src.config import settings


class DistillationPipeline:
    """Pipeline for generating labeled data using an LLM Teacher."""
    
    def __init__(
        self,
        teacher_url: Optional[str] = None,
        teacher_model: Optional[str] = None,
        confidence_threshold: float = 0.8,
        max_samples: int = 10000,
        batch_size: int = 10,
    ):
        self.teacher_url = teacher_url or settings.distillation.teacher_url
        self.teacher_model = teacher_model or settings.distillation.teacher_model
        self.confidence_threshold = confidence_threshold
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.system_prompt = settings.distillation.system_prompt
        
    async def _query_teacher(
        self,
        client: httpx.AsyncClient,
        text: str,
    ) -> Optional[dict]:
        """Query the LLM Teacher for a single text."""
        try:
            response = await client.post(
                self.teacher_url,
                json={
                    "model": self.teacher_model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Classifica questo testo:\n\n{text}"},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 50,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # Parse JSON response
            # Try to extract JSON from response (handle markdown code blocks)
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            parsed = json.loads(content)
            label = parsed.get("label", "").upper()
            confidence = float(parsed.get("confidence", 0.0))
            
            if label in ("AI", "NON_AI") and 0.0 <= confidence <= 1.0:
                return {
                    "label": 1 if label == "AI" else 0,
                    "confidence": confidence,
                }
                
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"Teacher query failed: {e}")
            
        return None
    
    async def _process_batch(
        self,
        client: httpx.AsyncClient,
        texts: list[str],
    ) -> list[tuple[str, dict]]:
        """Process a batch of texts concurrently."""
        tasks = [self._query_teacher(client, text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        valid_results = []
        for text, result in zip(texts, results):
            if result and result["confidence"] >= self.confidence_threshold:
                valid_results.append((text, result))
                
        return valid_results
    
    def load_unlabeled_texts(
        self,
        source_dir: Optional[Path] = None,
    ) -> list[str]:
        """Load unlabeled texts from directory."""
        if source_dir is None:
            source_dir = settings.data_dir / "unlabeled"
        
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Unlabeled data directory not found: {source_dir}\n"
                "Create this directory and add .txt files to label."
            )
        
        texts = []
        for txt_file in source_dir.glob("*.txt"):
            content = txt_file.read_text(encoding="utf-8").strip()
            if content:
                texts.append(content)
        
        if not texts:
            raise ValueError(f"No .txt files found in {source_dir}")
        
        logger.info(f"Loaded {len(texts)} unlabeled texts from {source_dir}")
        return texts
    
    async def run(
        self,
        texts: Optional[list[str]] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Run distillation pipeline.
        
        Args:
            texts: List of texts to label (or load from unlabeled/)
            output_path: Path to save labeled data
            
        Returns:
            Path to generated JSONL file
        """
        if texts is None:
            texts = self.load_unlabeled_texts()
        
        if output_path is None:
            output_path = settings.data_dir / "distilled.jsonl"
        
        # Limit to max_samples
        if len(texts) > self.max_samples:
            logger.info(f"Limiting to {self.max_samples} samples")
            texts = texts[:self.max_samples]
        
        labeled_data = []
        
        async with httpx.AsyncClient() as client:
            # Test connection
            try:
                await self._query_teacher(client, "Test connection")
                logger.info(f"Connected to LLM Teacher at {self.teacher_url}")
            except Exception as e:
                raise ConnectionError(f"Cannot connect to LLM Teacher: {e}")
            
            # Process in batches
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Distilling"):
                batch = texts[i:i + self.batch_size]
                results = await self._process_batch(client, batch)
                labeled_data.extend(results)
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)
        
        # Save to JSONL
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for text, result in labeled_data:
                item = {
                    "text": text,
                    "label": result["label"],
                    "confidence": result["confidence"],
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(labeled_data)} labeled samples to {output_path}")
        logger.info(
            f"Label distribution: "
            f"AI={sum(1 for _, r in labeled_data if r['label'] == 1)}, "
            f"NON_AI={sum(1 for _, r in labeled_data if r['label'] == 0)}"
        )
        
        return output_path


def run_distillation(
    texts: Optional[list[str]] = None,
    output_path: Optional[Path] = None,
    **kwargs,
) -> Path:
    """Synchronous wrapper for distillation pipeline."""
    pipeline = DistillationPipeline(**kwargs)
    return asyncio.run(pipeline.run(texts, output_path))

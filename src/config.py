"""
Pack-a-Punch Configuration Module

Centralized configuration using Pydantic for type safety and validation.
"""

from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data"
MODELS_DIR = PROJECT_ROOT / "src" / "models"


class ModelConfig(BaseModel):
    """Model configuration."""
    name: str = "DeepMount00/Italian-ModernBERT-base"
    max_length: int = 512
    num_labels: int = 2
    label_map: dict[int, str] = {0: "NON_AI", 1: "AI"}


class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = True
    save_steps: int = 500
    eval_steps: int = 100
    early_stopping_patience: int = 3


class DistillationConfig(BaseModel):
    """LLM Teacher distillation settings."""
    teacher_url: str = "http://localhost:1234/v1/chat/completions"
    teacher_model: str = "local-model"
    batch_size: int = 10
    confidence_threshold: float = 0.8
    max_samples: int = 10000
    system_prompt: str = """Sei un classificatore di testi. Devi determinare se il testo seguente parla di intelligenza artificiale (AI) o di altri argomenti (NON_AI).

Rispondi SOLO con un JSON nel formato: {"label": "AI" o "NON_AI", "confidence": 0.0-1.0}"""


class InferenceConfig(BaseModel):
    """Inference engine settings."""
    batch_size: int = 64
    num_sessions: int = 2  # Parallel ONNX sessions
    max_queue_size: int = 1000
    batch_timeout_ms: int = 50
    use_fp16: bool = True
    device: Literal["cuda", "cpu"] = "cuda"


class ServerConfig(BaseModel):
    """API server settings."""
    host: str = "0.0.0.0"
    port: int = 8080
    max_batch_size: int = 100
    workers: int = 1  # Single worker for GPU
    timeout: int = 60


class Settings(BaseSettings):
    """Main settings aggregator."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    
    # Paths
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR
    
    class Config:
        env_prefix = "PAP_"
        env_nested_delimiter = "__"


# Global settings instance
settings = Settings()

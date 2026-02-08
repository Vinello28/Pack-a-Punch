"""
Pack-a-Punch Configuration Module

Centralized configuration using Pydantic for type safety and validation.
Loads settings from config/model_config.yml when available.
"""

from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from .config_loader import (
    load_yaml_config,
    get_model_config,
    get_training_config,
    get_inference_config,
    get_server_config,
    get_distillation_config,
)


# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data"
MODELS_DIR = PROJECT_ROOT / "src" / "models"

# Load YAML config once at module level
_yaml_config = load_yaml_config()
_model_cfg = get_model_config(_yaml_config)
_training_cfg = get_training_config(_yaml_config)
_inference_cfg = get_inference_config(_yaml_config)
_server_cfg = get_server_config(_yaml_config)
_distillation_cfg = get_distillation_config(_yaml_config)


class ModelArchitecture(BaseModel):
    """Model architecture parameters for ONNX optimization."""
    num_heads: int = _model_cfg["architecture"]["num_heads"]
    hidden_size: int = _model_cfg["architecture"]["hidden_size"]
    model_type: str = _model_cfg["architecture"]["model_type"]


class ModelConfig(BaseModel):
    """Model configuration."""
    name: str = _model_cfg["name"]
    max_length: int = _model_cfg["max_length"]
    num_labels: int = _model_cfg["num_labels"]
    label_map: dict[int, str] = Field(default_factory=lambda: _model_cfg["label_map"])
    architecture: ModelArchitecture = Field(default_factory=ModelArchitecture)


class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    batch_size: int = _training_cfg["batch_size"]
    learning_rate: float = _training_cfg["learning_rate"]
    num_epochs: int = _training_cfg["num_epochs"]
    warmup_ratio: float = _training_cfg["warmup_ratio"]
    weight_decay: float = _training_cfg["weight_decay"]
    max_grad_norm: float = _training_cfg["max_grad_norm"]
    fp16: bool = _training_cfg["fp16"]
    save_steps: int = _training_cfg["save_steps"]
    eval_steps: int = _training_cfg["eval_steps"]
    early_stopping_patience: int = _training_cfg["early_stopping_patience"]


class DistillationConfig(BaseModel):
    """LLM Teacher distillation settings."""
    teacher_url: str = _distillation_cfg["teacher_url"]
    teacher_model: str = _distillation_cfg["teacher_model"]
    batch_size: int = _distillation_cfg["batch_size"]
    confidence_threshold: float = _distillation_cfg["confidence_threshold"]
    max_samples: int = _distillation_cfg["max_samples"]
    system_prompt: str = _distillation_cfg["system_prompt"]


class InferenceConfig(BaseModel):
    """Inference engine settings."""
    batch_size: int = _inference_cfg["batch_size"]
    num_sessions: int = _inference_cfg["num_sessions"]
    max_queue_size: int = _inference_cfg["max_queue_size"]
    batch_timeout_ms: int = _inference_cfg["batch_timeout_ms"]
    use_fp16: bool = _inference_cfg["use_fp16"]
    device: Literal["cuda", "cpu", "mps"] = _inference_cfg["device"]


class ServerConfig(BaseModel):
    """API server settings."""
    host: str = _server_cfg["host"]
    port: int = _server_cfg["port"]
    max_batch_size: int = _server_cfg["max_batch_size"]
    workers: int = _server_cfg["workers"]
    timeout: int = _server_cfg["timeout"]


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
    
    model_config = {
        "env_prefix": "PAP_",
        "env_nested_delimiter": "__",
    }


# Global settings instance
settings = Settings()

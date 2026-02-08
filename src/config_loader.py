"""
YAML Configuration Loader for Pack-a-Punch.

Loads model and training configuration from config/model_config.yml
with fallback to defaults if file is not found.
"""

from pathlib import Path
from typing import Any, Optional
import yaml
from loguru import logger


# Default config paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_FILE = CONFIG_DIR / "model_config.yml"


def load_yaml_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. Defaults to config/model_config.yml
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_FILE
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return {}
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config or {}
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config file: {e}")
        return {}


def get_model_config(config: Optional[dict] = None) -> dict[str, Any]:
    """
    Extract model configuration with defaults.
    
    Args:
        config: Full config dict (loads from file if None)
        
    Returns:
        Model configuration dictionary
    """
    if config is None:
        config = load_yaml_config()
    
    defaults = {
        "name": "dbmdz/bert-base-italian-xxl-cased",
        "max_length": 512,
        "num_labels": 2,
        "label_map": {0: "NON_AI", 1: "AI"},
        "architecture": {
            "num_heads": 12,
            "hidden_size": 768,
            "model_type": "bert",
        },
    }
    
    model_config = config.get("model", {})
    
    # Merge with defaults
    result = {**defaults, **model_config}
    
    # Ensure label_map keys are integers
    if "label_map" in model_config:
        result["label_map"] = {
            int(k): v for k, v in model_config["label_map"].items()
        }
    
    # Merge architecture
    if "architecture" in model_config:
        result["architecture"] = {
            **defaults["architecture"],
            **model_config["architecture"],
        }
    
    return result


def get_training_config(config: Optional[dict] = None) -> dict[str, Any]:
    """Extract training configuration with defaults."""
    if config is None:
        config = load_yaml_config()
    
    defaults = {
        "batch_size": 32,
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "fp16": True,
        "save_steps": 500,
        "eval_steps": 100,
        "early_stopping_patience": 3,
    }
    
    return {**defaults, **config.get("training", {})}


def get_inference_config(config: Optional[dict] = None) -> dict[str, Any]:
    """Extract inference configuration with defaults."""
    if config is None:
        config = load_yaml_config()
    
    defaults = {
        "batch_size": 64,
        "num_sessions": 2,
        "max_queue_size": 1000,
        "batch_timeout_ms": 50,
        "use_fp16": True,
        "device": "cuda",
    }
    
    return {**defaults, **config.get("inference", {})}


def get_server_config(config: Optional[dict] = None) -> dict[str, Any]:
    """Extract server configuration with defaults."""
    if config is None:
        config = load_yaml_config()
    
    defaults = {
        "host": "0.0.0.0",
        "port": 8080,
        "max_batch_size": 100,
        "workers": 1,
        "timeout": 60,
    }
    
    return {**defaults, **config.get("server", {})}


def get_distillation_config(config: Optional[dict] = None) -> dict[str, Any]:
    """Extract distillation configuration with defaults."""
    if config is None:
        config = load_yaml_config()
    
    defaults = {
        "teacher_url": "http://localhost:1234/v1/chat/completions",
        "teacher_model": "local-model",
        "batch_size": 10,
        "confidence_threshold": 0.8,
        "max_samples": 10000,
        "system_prompt": """Sei un classificatore di testi. Devi determinare se il testo seguente parla di intelligenza artificiale (AI) o di altri argomenti (NON_AI).

Rispondi SOLO con un JSON nel formato: {"label": "AI" o "NON_AI", "confidence": 0.0-1.0}""",
    }
    
    return {**defaults, **config.get("distillation", {})}

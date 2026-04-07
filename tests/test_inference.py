"""
Tests for inference module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Test configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestInferenceEngine:
    """Tests for InferenceEngine class."""
    
    def test_tokenize(self):
        """Test tokenization produces correct shape."""
        from src.config import settings
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(settings.model.name)
        
        texts = ["Testo di prova", "Altro testo"]
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=settings.model.max_length,
            padding="max_length",
            return_tensors="np",
        )
        
        assert encodings["input_ids"].shape == (2, settings.model.max_length)
        assert encodings["attention_mask"].shape == (2, settings.model.max_length)
    
    def test_softmax_calculation(self):
        """Test softmax produces valid probabilities."""
        # Simulate logits
        logits = np.array([
            [2.0, 1.0, 0.5, 0.1, -0.3, 0.8, 0.2, -0.5],
            [0.5, 1.5, -0.2, 0.3, 0.7, -0.1, 1.0, 0.4],
            [-1.0, 2.0, 0.3, -0.5, 0.1, 0.6, -0.2, 1.2],
        ])
        
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Check valid probabilities
        assert np.allclose(probs.sum(axis=-1), 1.0)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
    
    def test_label_mapping(self):
        """Test label mapping correctness."""
        from src.config import settings
        
        assert len(settings.model.label_map) == 8
        assert settings.model.label_map[0] == "AI Research"
        assert settings.model.label_map[7] == "Virtual Assistants"


class TestDataset:
    """Tests for dataset module."""
    
    def test_load_dataset_from_jsonl(self, tmp_path):
        """Test JSONL loading."""
        from src.training.dataset import load_dataset_from_jsonl

        # Create test file with multi-class labels
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text(
            '{"text": "Test 1", "label": 0}\n'
            '{"text": "Test 2", "label": 3}\n'
            '{"text": "Test 3", "label": "Data Science"}\n'
        )

        texts, labels = load_dataset_from_jsonl(jsonl_path)

        assert len(texts) == 3
        assert labels == [0, 3, 2]
    
    def test_load_dataset_from_txt(self, tmp_path):
        """Test TXT directory loading."""
        from src.training.dataset import load_dataset_from_txt

        # Create test directories matching label slugs
        dir_a = tmp_path / "ai_research"
        dir_b = tmp_path / "data_science"
        dir_a.mkdir()
        dir_b.mkdir()

        (dir_a / "sample1.txt").write_text("AI research text")
        (dir_b / "sample2.txt").write_text("Data science text")

        texts, labels = load_dataset_from_txt(tmp_path)

        assert len(texts) == 2
        assert 0 in labels  # AI Research
        assert 2 in labels  # Data Science


class TestServer:
    """Tests for FastAPI server."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create mock inference engine."""
        engine = Mock()
        engine.predict_parallel = Mock(return_value=[
            {"label": "AI Research", "confidence": 0.95},
        ])
        engine.get_stats = Mock(return_value={"status": "test"})
        return engine
    
    def test_classify_request_validation(self):
        """Test request validation."""
        from src.inference.server import ClassifyRequest
        
        # Valid request
        req = ClassifyRequest(texts=["test"])
        assert len(req.texts) == 1
        
        # Empty texts should fail
        with pytest.raises(ValueError):
            ClassifyRequest(texts=[])
    
    def test_prediction_response(self):
        """Test response model."""
        from src.inference.server import Prediction, ClassifyResponse
        
        pred = Prediction(label="AI Research", confidence=0.95)
        assert pred.label == "AI Research"
        assert pred.confidence == 0.95
        
        response = ClassifyResponse(
            predictions=[pred],
            processing_time_ms=42.5,
        )
        assert len(response.predictions) == 1


class TestConfig:
    """Tests for configuration module."""
    
    def test_settings_defaults(self):
        """Test default settings are valid."""
        from src.config import settings
        
        assert settings.model.max_length == 3072
        assert settings.model.num_labels == 8
        assert settings.inference.num_sessions >= 1
        assert settings.server.port > 0
    
    def test_paths_exist(self):
        """Test path configuration."""
        from src.config import DATA_DIR, MODELS_DIR
        
        assert DATA_DIR.name == "data"
        assert MODELS_DIR.name == "models"

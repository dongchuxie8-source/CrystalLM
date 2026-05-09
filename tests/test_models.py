"""
Tests for CrystalLM Models.

This module contains unit tests for the model classes.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestT5Translator:
    """Test cases for T5Translator."""
    
    @pytest.fixture
    def translator(self):
        """Create translator instance (uses small model for testing)."""
        try:
            from crystallm.models.t5_translator import T5Translator
            # Use small model for faster testing
            return T5Translator("google/t5-v1_1-small", device="cpu")
        except ImportError:
            pytest.skip("transformers not installed")
    
    def test_initialization(self, translator):
        """Test translator initialization."""
        assert translator.model is not None
        assert translator.tokenizer is not None
        assert translator.device == "cpu"
    
    def test_translate_single(self, translator):
        """Test single text translation."""
        input_text = "translate structure to text: [CRYSTAL] [META] MOF Test [/CRYSTAL]"
        output = translator.translate(input_text, max_length=50)
        
        assert isinstance(output, str)
        assert len(output) > 0
    
    def test_translate_batch(self, translator):
        """Test batch translation."""
        inputs = [
            "translate structure to text: [CRYSTAL] [META] MOF Test1 [/CRYSTAL]",
            "translate structure to text: [CRYSTAL] [META] MOF Test2 [/CRYSTAL]"
        ]
        outputs = translator.translate(inputs, max_length=50)
        
        assert isinstance(outputs, list)
        assert len(outputs) == 2
    
    def test_translate_with_sampling(self, translator):
        """Test translation with sampling."""
        input_text = "translate structure to text: [CRYSTAL] [META] MOF Test [/CRYSTAL]"
        output = translator.translate(
            input_text,
            max_length=50,
            do_sample=True,
            temperature=0.8
        )
        
        assert isinstance(output, str)
    
    def test_translate_multiple_sequences(self, translator):
        """Test generating multiple sequences."""
        input_text = "translate structure to text: [CRYSTAL] [META] MOF Test [/CRYSTAL]"
        outputs = translator.translate(
            input_text,
            max_length=50,
            num_return_sequences=3,
            num_beams=3
        )
        
        assert isinstance(outputs, list)
        assert len(outputs) == 3
    
    def test_to_device(self, translator):
        """Test moving model to device."""
        translator.to("cpu")
        assert translator.device == "cpu"
    
    def test_train_eval_mode(self, translator):
        """Test switching between train and eval modes."""
        translator.train_mode()
        assert translator.model.training
        
        translator.eval_mode()
        assert not translator.model.training


class TestMetrics:
    """Test cases for evaluation metrics."""
    
    def test_compute_bleu(self):
        """Test BLEU score computation."""
        from crystallm.evaluation.metrics import compute_bleu
        
        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]
        
        scores = compute_bleu(predictions, references)
        
        assert 'bleu_1' in scores
        assert 'bleu_4' in scores
        assert scores['bleu_1'] > 0
    
    def test_compute_exact_match(self):
        """Test exact match computation."""
        from crystallm.evaluation.metrics import compute_exact_match
        
        predictions = ["hello world", "foo bar"]
        references = ["hello world", "foo baz"]
        
        em = compute_exact_match(predictions, references)
        
        assert em == 0.5
    
    def test_compute_metrics_s2t(self):
        """Test computing all S2T metrics."""
        from crystallm.evaluation.metrics import compute_metrics
        
        predictions = ["MOF-5 is a zinc-based framework."]
        references = ["MOF-5 is a zinc-based metal-organic framework."]
        
        metrics = compute_metrics(predictions, references, task="s2t")
        
        assert 'bleu_4' in metrics
        assert 'rouge1' in metrics
    
    def test_compute_metrics_t2s(self):
        """Test computing all T2S metrics."""
        from crystallm.evaluation.metrics import compute_metrics
        
        predictions = ["[CRYSTAL] [META] MOF Test [/CRYSTAL]"]
        references = ["[CRYSTAL] [META] MOF Test [/CRYSTAL]"]
        
        metrics = compute_metrics(predictions, references, task="t2s")
        
        assert 'exact_match' in metrics
        assert 'validity_rate' in metrics


class TestTrainingConfig:
    """Test cases for training configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from crystallm.training.config import TrainingConfig
        
        config = TrainingConfig()
        
        assert config.num_epochs == 8
        assert config.batch_size == 16
        assert config.learning_rate == 1e-4
    
    def test_custom_config(self):
        """Test custom configuration values."""
        from crystallm.training.config import TrainingConfig
        
        config = TrainingConfig(
            num_epochs=10,
            batch_size=32,
            learning_rate=5e-5
        )
        
        assert config.num_epochs == 10
        assert config.batch_size == 32
        assert config.learning_rate == 5e-5
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        from crystallm.training.config import TrainingConfig
        
        config = TrainingConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'num_epochs' in config_dict
        assert 'learning_rate' in config_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

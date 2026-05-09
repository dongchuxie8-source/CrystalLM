"""
Tests for CrystalLM Dataset.

This module contains unit tests for the dataset classes.
"""

import pytest
import json
import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCrystalLMDataset:
    """Test cases for CrystalLMDataset."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return [
            {
                "hssr": "[CRYSTAL] [META] MOF MOF-5 [COMP] Zn4O(C8H4O4)3 [/CRYSTAL]",
                "description": "MOF-5 is a zinc-based metal-organic framework."
            },
            {
                "hssr": "[CRYSTAL] [META] MOF HKUST-1 [COMP] Cu3(C9H3O6)2 [/CRYSTAL]",
                "description": "HKUST-1 is a copper-based MOF with paddlewheel clusters."
            }
        ]
    
    @pytest.fixture
    def temp_data_file(self, sample_data):
        """Create temporary data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            return f.name
    
    def test_dataset_loading(self, temp_data_file):
        """Test dataset loading from file."""
        try:
            from transformers import T5Tokenizer
            from crystallm.data.dataset import CrystalLMDataset
            
            tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base")
            dataset = CrystalLMDataset(
                temp_data_file,
                tokenizer,
                max_length=128,
                task="s2t"
            )
            
            assert len(dataset) == 2
        except ImportError:
            pytest.skip("transformers not installed")
        finally:
            os.unlink(temp_data_file)
    
    def test_dataset_getitem_s2t(self, temp_data_file):
        """Test getting item for S2T task."""
        try:
            from transformers import T5Tokenizer
            from crystallm.data.dataset import CrystalLMDataset
            
            tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base")
            dataset = CrystalLMDataset(
                temp_data_file,
                tokenizer,
                max_length=128,
                task="s2t"
            )
            
            item = dataset[0]
            
            assert 'input_ids' in item
            assert 'attention_mask' in item
            assert 'labels' in item
            
            # Check shapes
            assert item['input_ids'].shape[0] == 128
            assert item['attention_mask'].shape[0] == 128
            assert item['labels'].shape[0] == 128
        except ImportError:
            pytest.skip("transformers not installed")
        finally:
            os.unlink(temp_data_file)
    
    def test_dataset_getitem_t2s(self, temp_data_file):
        """Test getting item for T2S task."""
        try:
            from transformers import T5Tokenizer
            from crystallm.data.dataset import CrystalLMDataset
            
            tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base")
            dataset = CrystalLMDataset(
                temp_data_file,
                tokenizer,
                max_length=128,
                task="t2s"
            )
            
            item = dataset[0]
            
            assert 'input_ids' in item
            assert 'attention_mask' in item
            assert 'labels' in item
        except ImportError:
            pytest.skip("transformers not installed")
        finally:
            os.unlink(temp_data_file)
    
    def test_invalid_data_format(self):
        """Test handling of invalid data format."""
        try:
            from transformers import T5Tokenizer
            from crystallm.data.dataset import CrystalLMDataset
            
            # Create invalid data (missing required keys)
            invalid_data = [{"hssr": "test"}]  # Missing 'description'
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(invalid_data, f)
                temp_file = f.name
            
            tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base")
            
            with pytest.raises(ValueError):
                CrystalLMDataset(temp_file, tokenizer, task="s2t")
            
            os.unlink(temp_file)
        except ImportError:
            pytest.skip("transformers not installed")


class TestCrystalLMDataModule:
    """Test cases for CrystalLMDataModule."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return [
            {
                "hssr": "[CRYSTAL] [META] MOF Test [/CRYSTAL]",
                "description": "Test description."
            }
        ]
    
    @pytest.fixture
    def temp_files(self, sample_data):
        """Create temporary train/val/test files."""
        files = {}
        for split in ['train', 'val', 'test']:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(sample_data, f)
                files[split] = f.name
        return files
    
    def test_data_module_setup(self, temp_files):
        """Test data module setup."""
        try:
            from transformers import T5Tokenizer
            from crystallm.data.dataset import CrystalLMDataModule
            
            tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base")
            
            data_module = CrystalLMDataModule(
                train_file=temp_files['train'],
                val_file=temp_files['val'],
                test_file=temp_files['test'],
                tokenizer=tokenizer,
                max_length=128,
                task="s2t",
                batch_size=1
            )
            
            data_module.setup()
            
            assert data_module.train_dataset is not None
            assert data_module.val_dataset is not None
            assert data_module.test_dataset is not None
        except ImportError:
            pytest.skip("transformers not installed")
        finally:
            for f in temp_files.values():
                os.unlink(f)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
PyTorch Dataset classes for CrystalLM.
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import random


class CrystalLMDataset(Dataset):
    """
    Dataset for CrystalLM structure-text translation.
    
    Supports both S2T (structure-to-text) and T2S (text-to-structure) tasks.
    """
    
    def __init__(
        self,
        data_file: str,
        tokenizer,
        max_length: int = 512,
        task: str = "s2t",
        augment: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            data_file: Path to JSON data file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            task: Task type ("s2t" or "t2s")
            augment: Whether to apply data augmentation
        """
        self.data = self.load_data(data_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        self.augment = augment
        
        print(f"Loaded {len(self.data)} examples for {task} task")
    
    def load_data(self, data_file: str) -> List[Dict]:
        """Load data from JSON file."""
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate data format
        required_keys = ['hssr', 'description']
        for i, item in enumerate(data):
            if not all(k in item for k in required_keys):
                raise ValueError(f"Item {i} missing required keys: {required_keys}")
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.
        
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        item = self.data[idx]
        
        # Get input and target based on task
        if self.task == "s2t":
            # Structure to Text
            input_text = item['hssr']
            target_text = item['description']
            task_prefix = "translate structure to text: "
        else:
            # Text to Structure
            input_text = item['description']
            target_text = item['hssr']
            task_prefix = "translate text to structure: "
        
        # Apply augmentation if enabled
        if self.augment:
            input_text = self.augment_text(input_text, self.task)
        
        # Add task prefix
        input_text = task_prefix + input_text
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # Prepare labels (replace padding token id with -100)
        labels = targets['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }
    
    def augment_text(self, text: str, task: str) -> str:
        """
        Apply data augmentation.
        
        For S2T: Randomly reorder non-essential HSSR fields
        For T2S: Apply synonym replacement and paraphrasing
        """
        if task == "s2t":
            # For HSSR, we can reorder some fields
            # This is a simple placeholder - actual implementation would be more sophisticated
            return text
        else:
            # For text, we could apply synonym replacement
            # This is a placeholder - actual implementation would use NLP techniques
            return text
    
    def get_collate_fn(self):
        """Return collate function for DataLoader."""
        def collate_fn(batch):
            """Collate batch of examples."""
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
            labels = torch.stack([item['labels'] for item in batch])
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
        
        return collate_fn


class CrystalLMDataModule:
    """
    Data module for managing train/val/test datasets.
    """
    
    def __init__(
        self,
        train_file: str,
        val_file: str,
        test_file: str,
        tokenizer,
        max_length: int = 512,
        task: str = "s2t",
        batch_size: int = 16,
        num_workers: int = 4
    ):
        """
        Initialize data module.
        
        Args:
            train_file: Path to training data
            val_file: Path to validation data
            test_file: Path to test data
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            task: Task type ("s2t" or "t2s")
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
        """
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup datasets."""
        self.train_dataset = CrystalLMDataset(
            self.train_file,
            self.tokenizer,
            self.max_length,
            self.task,
            augment=True
        )
        
        self.val_dataset = CrystalLMDataset(
            self.val_file,
            self.tokenizer,
            self.max_length,
            self.task,
            augment=False
        )
        
        self.test_dataset = CrystalLMDataset(
            self.test_file,
            self.tokenizer,
            self.max_length,
            self.task,
            augment=False
        )
    
    def train_dataloader(self):
        """Get training DataLoader."""
        from torch.utils.data import DataLoader
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.get_collate_fn(),
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Get validation DataLoader."""
        from torch.utils.data import DataLoader
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.get_collate_fn(),
            pin_memory=True
        )
    
    def test_dataloader(self):
        """Get test DataLoader."""
        from torch.utils.data import DataLoader
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset.get_collate_fn(),
            pin_memory=True
        )


if __name__ == "__main__":
    # Example usage
    from transformers import T5Tokenizer
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base")
    
    # Create dummy data for testing
    dummy_data = [
        {
            "hssr": "[CRYSTAL] [META] MOF MOF-5 [COMP] Zn4O(C8H4O4)3 [/CRYSTAL]",
            "description": "MOF-5 is a zinc-based metal-organic framework."
        }
    ]
    
    with open("dummy_data.json", "w") as f:
        json.dump(dummy_data, f)
    
    # Create dataset
    dataset = CrystalLMDataset(
        "dummy_data.json",
        tokenizer,
        max_length=128,
        task="s2t"
    )
    
    # Get an example
    example = dataset[0]
    print("Input IDs shape:", example['input_ids'].shape)
    print("Attention mask shape:", example['attention_mask'].shape)
    print("Labels shape:", example['labels'].shape)
    
    # Decode to verify
    input_text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
    print(f"\nInput text: {input_text}")

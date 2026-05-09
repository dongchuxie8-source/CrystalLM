"""
Training configuration for CrystalLM.

This module defines configuration classes for model training,
including hyperparameters, paths, and optimization settings.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import yaml
import json


@dataclass
class TrainingConfig:
    """
    Configuration for CrystalLM training.
    
    Attributes:
        output_dir: Directory to save model checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size per device
        eval_batch_size: Evaluation batch size per device
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Initial learning rate
        weight_decay: Weight decay for AdamW optimizer
        warmup_steps: Number of warmup steps
        max_grad_norm: Maximum gradient norm for clipping
        fp16: Whether to use mixed precision training
        logging_steps: Log every N steps
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        save_total_limit: Maximum number of checkpoints to keep
        early_stopping_patience: Patience for early stopping
        early_stopping_threshold: Minimum improvement threshold
        seed: Random seed for reproducibility
    """
    
    # Output settings
    output_dir: str = "models/crystallm"
    
    # Training hyperparameters
    num_epochs: int = 8
    batch_size: int = 16
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Mixed precision
    fp16: bool = True
    bf16: bool = False
    
    # Logging and evaluation
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 2000
    save_total_limit: int = 3
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Reproducibility
    seed: int = 42
    
    # Distributed training
    local_rank: int = -1
    
    # DeepSpeed
    deepspeed: Optional[str] = None
    
    @property
    def total_steps(self) -> int:
        """Calculate total training steps (placeholder, needs dataset size)."""
        return self.num_epochs * 1000  # Placeholder
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            TrainingConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Flatten nested config if needed
        if 'training' in config_dict:
            config_dict = {**config_dict.get('training', {}), **config_dict}
            del config_dict['training']
        
        return cls(**{k: v for k, v in config_dict.items() 
                     if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, json_path: str) -> "TrainingConfig":
        """
        Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            TrainingConfig instance
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**{k: v for k, v in config_dict.items() 
                     if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'output_dir': self.output_dir,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'eval_batch_size': self.eval_batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'max_grad_norm': self.max_grad_norm,
            'fp16': self.fp16,
            'bf16': self.bf16,
            'logging_steps': self.logging_steps,
            'eval_steps': self.eval_steps,
            'save_steps': self.save_steps,
            'save_total_limit': self.save_total_limit,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_threshold': self.early_stopping_threshold,
            'seed': self.seed,
        }
    
    def save(self, path: str):
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration (YAML or JSON based on extension)
        """
        config_dict = self.to_dict()
        
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)


@dataclass
class ModelConfig:
    """
    Configuration for model architecture.
    
    Attributes:
        model_name: HuggingFace model name or path
        max_length: Maximum sequence length
        num_beams: Number of beams for generation
    """
    
    model_name: str = "google/t5-v1_1-large"
    max_length: int = 512
    num_beams: int = 5
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if 'model' in config_dict:
            config_dict = config_dict['model']
        
        return cls(**{k: v for k, v in config_dict.items() 
                     if k in cls.__dataclass_fields__})


@dataclass
class DataConfig:
    """
    Configuration for data processing.
    
    Attributes:
        train_file: Path to training data
        val_file: Path to validation data
        test_file: Path to test data
        max_samples: Maximum number of samples (None for all)
        num_workers: Number of data loading workers
    """
    
    train_file: str = "data/train.json"
    val_file: str = "data/val.json"
    test_file: str = "data/test.json"
    max_samples: Optional[int] = None
    num_workers: int = 4
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DataConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if 'data' in config_dict:
            config_dict = config_dict['data']
        
        return cls(**{k: v for k, v in config_dict.items() 
                     if k in cls.__dataclass_fields__})


if __name__ == "__main__":
    # Example usage
    config = TrainingConfig(
        output_dir="models/s2t_t5large",
        num_epochs=8,
        batch_size=16,
        learning_rate=1e-4
    )
    
    print("Training Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Save to file
    config.save("example_config.yaml")
    print("\nConfiguration saved to example_config.yaml")

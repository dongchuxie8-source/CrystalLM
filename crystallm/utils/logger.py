"""
Logging utilities for CrystalLM.

This module provides logging configuration and utilities
for consistent logging across the project.
"""

import logging
import sys
from typing import Optional
from datetime import datetime
import os


def setup_logger(
    name: str = "crystallm",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup and configure a logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Create directory if needed
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "crystallm") -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set up basic configuration
    if not logger.handlers:
        setup_logger(name)
    
    return logger


class TrainingLogger:
    """
    Logger for tracking training progress.
    
    Provides methods for logging training metrics, checkpoints,
    and other training-related information.
    """
    
    def __init__(
        self,
        output_dir: str,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize training logger.
        
        Args:
            output_dir: Directory for log files
            experiment_name: Name of the experiment
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        # Setup file logger
        log_file = os.path.join(output_dir, f"{experiment_name}.log")
        self.logger = setup_logger(
            name=f"crystallm.{experiment_name}",
            log_file=log_file
        )
        
        # Metrics history
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        **kwargs
    ):
        """
        Log epoch results.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            learning_rate: Current learning rate
            **kwargs: Additional metrics
        """
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['learning_rate'].append(learning_rate)
        
        # Log additional metrics
        for key, value in kwargs.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
        
        # Format message
        msg = f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={learning_rate:.2e}"
        for key, value in kwargs.items():
            msg += f", {key}={value:.4f}"
        
        self.logger.info(msg)
    
    def log_step(self, step: int, loss: float, **kwargs):
        """
        Log training step.
        
        Args:
            step: Step number
            loss: Current loss
            **kwargs: Additional metrics
        """
        msg = f"Step {step}: loss={loss:.4f}"
        for key, value in kwargs.items():
            msg += f", {key}={value:.4f}"
        
        self.logger.debug(msg)
    
    def log_checkpoint(self, checkpoint_path: str, metrics: dict):
        """
        Log checkpoint save.
        
        Args:
            checkpoint_path: Path to saved checkpoint
            metrics: Metrics at checkpoint
        """
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        self.logger.info(f"Metrics: {metrics}")
    
    def log_evaluation(self, metrics: dict, split: str = "test"):
        """
        Log evaluation results.
        
        Args:
            metrics: Evaluation metrics
            split: Data split name
        """
        self.logger.info(f"Evaluation on {split}:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
    
    def save_metrics(self):
        """Save metrics history to file."""
        import json
        
        metrics_file = os.path.join(
            self.output_dir, 
            f"{self.experiment_name}_metrics.json"
        )
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.logger.info(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    # Example usage
    logger = setup_logger("test", level=logging.DEBUG)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Training logger example
    training_logger = TrainingLogger("logs", "test_experiment")
    training_logger.log_epoch(1, 0.5, 0.4, 1e-4, bleu=0.35)
    training_logger.log_epoch(2, 0.4, 0.35, 5e-5, bleu=0.40)
    training_logger.save_metrics()

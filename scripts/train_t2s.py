#!/usr/bin/env python
"""
Train Text-to-Structure (T2S) model.

This script trains a T5 model for translating natural language
descriptions to crystal structure representations (HSSR).
"""

import argparse
import os
import sys
import logging
from datetime import datetime

import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crystallm.data.dataset import CrystalLMDataModule

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train T2S model")
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='google/t5-v1_1-large',
                       help='Pretrained model name or path')
    
    # Data arguments
    parser.add_argument('--train_file', type=str, required=True,
                       help='Path to training data JSON')
    parser.add_argument('--val_file', type=str, required=True,
                       help='Path to validation data JSON')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for model and logs')
    parser.add_argument('--num_train_epochs', type=int, default=8,
                       help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=16,
                       help='Training batch size per device')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16,
                       help='Evaluation batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Number of warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--logging_steps', type=int, default=100,
                       help='Log every N steps')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluate every N steps')
    parser.add_argument('--save_steps', type=int, default=2000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--fp16', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log configuration
    logger.info("=" * 50)
    logger.info("CrystalLM T2S Training")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Train file: {args.train_file}")
    logger.info(f"Val file: {args.val_file}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Epochs: {args.num_train_epochs}")
    logger.info(f"Batch size: {args.per_device_train_batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("=" * 50)
    
    # Load tokenizer and model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Setup data - Note: task="t2s" for text-to-structure
    logger.info("Loading datasets...")
    data_module = CrystalLMDataModule(
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.val_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        task="t2s",  # Text-to-Structure task
        batch_size=args.per_device_train_batch_size
    )
    data_module.setup()
    
    logger.info(f"Training samples: {len(data_module.train_dataset)}")
    logger.info(f"Validation samples: {len(data_module.val_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=args.fp16,
        report_to="tensorboard",
        save_total_limit=3,
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_module.train_dataset,
        eval_dataset=data_module.val_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )
    
    # Train
    logger.info("Starting training...")
    start_time = datetime.now()
    
    train_result = trainer.train()
    
    end_time = datetime.now()
    training_time = end_time - start_time
    logger.info(f"Training completed in {training_time}")
    
    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    logger.info("=" * 50)
    logger.info("Training completed!")
    logger.info(f"Best model saved to: {args.output_dir}")
    logger.info(f"Final eval loss: {eval_metrics.get('eval_loss', 'N/A')}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

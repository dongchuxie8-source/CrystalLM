"""
CrystalLM Trainer.

This module implements the training loop for CrystalLM models,
including support for distributed training, mixed precision,
and early stopping.
"""

import os
import time
import logging
from typing import Optional, Dict, Any, Callable
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from crystallm.training.config import TrainingConfig

logger = logging.getLogger(__name__)


class CrystalLMTrainer:
    """
    Trainer for CrystalLM models.
    
    Handles the training loop, evaluation, checkpointing,
    and early stopping for structure-text translation models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        config: TrainingConfig,
        tokenizer=None,
        compute_metrics: Optional[Callable] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration
            tokenizer: Tokenizer for decoding (optional)
            compute_metrics: Function to compute metrics (optional)
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup data loaders
        self.train_dataloader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_dataloader = self._create_dataloader(val_dataset, shuffle=False)
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
    
    def _create_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        """Create DataLoader for dataset."""
        batch_size = self.config.batch_size if shuffle else self.config.eval_batch_size
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            collate_fn=getattr(dataset, 'get_collate_fn', lambda: None)()
        )
    
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer."""
        # Separate parameters for weight decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        num_training_steps = (
            len(self.train_dataloader) * self.config.num_epochs 
            // self.config.gradient_accumulation_steps
        )
        
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train(self) -> Dict[str, Any]:
        """
        Run training loop.
        
        Returns:
            Dictionary with training results
        """
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Train one epoch
            train_loss = self._train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self._validate()
            self.val_losses.append(val_loss)
            
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
            )
            
            # Check for improvement
            if val_loss < self.best_val_loss - self.config.early_stopping_threshold:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best_model")
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"No improvement. Patience: {self.patience_counter}/{self.config.early_stopping_patience}")
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info("Early stopping triggered!")
                break
            
            # Save periodic checkpoint
            if (epoch + 1) % 2 == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch + 1}")
        
        # Save final model
        self._save_checkpoint("final_model")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 60:.2f} minutes")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time
        }
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}",
            leave=False
        )
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.config.fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                avg_loss = total_loss / num_batches
                logger.debug(f"Step {self.global_step}: loss = {avg_loss:.4f}")
        
        return total_loss / num_batches
    
    def _validate(self) -> float:
        """
        Run validation.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.config.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _save_checkpoint(self, name: str):
        """
        Save model checkpoint.
        
        Args:
            name: Checkpoint name
        """
        save_path = os.path.join(self.config.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        
        # Save tokenizer if available
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path)
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(state, os.path.join(save_path, 'training_state.pt'))
        
        logger.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        # Load model
        self.model = self.model.from_pretrained(checkpoint_path)
        self.model.to(self.device)
        
        # Load training state
        state_path = os.path.join(checkpoint_path, 'training_state.pt')
        if os.path.exists(state_path):
            state = torch.load(state_path)
            self.global_step = state['global_step']
            self.epoch = state['epoch']
            self.best_val_loss = state['best_val_loss']
            self.train_losses = state['train_losses']
            self.val_losses = state['val_losses']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def evaluate(self, test_dataset=None) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_dataset: Test dataset (optional, uses val_dataset if None)
            
        Returns:
            Dictionary of evaluation metrics
        """
        dataset = test_dataset if test_dataset is not None else self.val_dataset
        dataloader = self._create_dataloader(dataset, shuffle=False)
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                
                # Generate predictions if tokenizer available
                if self.tokenizer is not None:
                    generated = self.model.generate(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        max_length=512,
                        num_beams=5
                    )
                    
                    predictions = self.tokenizer.batch_decode(
                        generated, 
                        skip_special_tokens=True
                    )
                    all_predictions.extend(predictions)
                    
                    # Decode labels
                    labels = batch['labels'].clone()
                    labels[labels == -100] = self.tokenizer.pad_token_id
                    references = self.tokenizer.batch_decode(
                        labels, 
                        skip_special_tokens=True
                    )
                    all_labels.extend(references)
        
        metrics = {'eval_loss': total_loss / len(dataloader)}
        
        # Compute additional metrics if function provided
        if self.compute_metrics is not None and all_predictions:
            additional_metrics = self.compute_metrics(all_predictions, all_labels)
            metrics.update(additional_metrics)
        
        return metrics


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("CrystalLMTrainer module loaded successfully.")
    print("Use with CrystalLMDataset and T5ForConditionalGeneration for training.")

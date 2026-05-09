"""
Evaluator for CrystalLM models.

This module provides a high-level interface for evaluating
trained models on test datasets.
"""

import json
import logging
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import torch

from crystallm.evaluation.metrics import compute_metrics, print_metrics

logger = logging.getLogger(__name__)


class CrystalLMEvaluator:
    """
    Evaluator for CrystalLM models.
    
    Handles model inference and metric computation for
    structure-text translation evaluation.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model (T5ForConditionalGeneration or similar)
            tokenizer: Tokenizer for the model
            device: Device to run evaluation on
        """
        self.model = model
        self.tokenizer = tokenizer
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Evaluator initialized on {self.device}")
    
    def generate(
        self,
        input_texts: List[str],
        max_length: int = 512,
        num_beams: int = 5,
        batch_size: int = 8,
        **generate_kwargs
    ) -> List[str]:
        """
        Generate outputs for input texts.
        
        Args:
            input_texts: List of input texts
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            batch_size: Batch size for inference
            **generate_kwargs: Additional arguments for model.generate()
            
        Returns:
            List of generated texts
        """
        all_outputs = []
        
        for i in tqdm(range(0, len(input_texts), batch_size), desc="Generating"):
            batch = input_texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    **generate_kwargs
                )
            
            # Decode
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_outputs.extend(decoded)
        
        return all_outputs
    
    def evaluate(
        self,
        test_data: List[Dict],
        task: str = "s2t",
        max_length: int = 512,
        num_beams: int = 5,
        batch_size: int = 8
    ) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: List of test examples with 'hssr' and 'description' keys
            task: Task type ("s2t" or "t2s")
            max_length: Maximum generation length
            num_beams: Number of beams
            batch_size: Batch size
            
        Returns:
            Dictionary with metrics and predictions
        """
        logger.info(f"Evaluating {len(test_data)} examples for {task} task")
        
        # Prepare inputs and references
        if task == "s2t":
            inputs = [f"translate structure to text: {item['hssr']}" for item in test_data]
            references = [item['description'] for item in test_data]
        else:
            inputs = [f"translate text to structure: {item['description']}" for item in test_data]
            references = [item['hssr'] for item in test_data]
        
        # Generate predictions
        predictions = self.generate(
            inputs,
            max_length=max_length,
            num_beams=num_beams,
            batch_size=batch_size
        )
        
        # Compute metrics
        metrics = compute_metrics(predictions, references, task=task)
        
        # Print results
        print_metrics(metrics, task=task)
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'references': references
        }
    
    def evaluate_from_file(
        self,
        test_file: str,
        task: str = "s2t",
        output_file: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate model on test file.
        
        Args:
            test_file: Path to test JSON file
            task: Task type
            output_file: Path to save results (optional)
            **kwargs: Additional arguments for evaluate()
            
        Returns:
            Evaluation results
        """
        # Load test data
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        logger.info(f"Loaded {len(test_data)} examples from {test_file}")
        
        # Evaluate
        results = self.evaluate(test_data, task=task, **kwargs)
        
        # Save results if output file specified
        if output_file:
            # Prepare output
            output = {
                'metrics': results['metrics'],
                'num_examples': len(test_data),
                'task': task,
                'examples': []
            }
            
            # Add sample predictions
            for i in range(min(100, len(results['predictions']))):
                output['examples'].append({
                    'prediction': results['predictions'][i],
                    'reference': results['references'][i]
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_file}")
        
        return results
    
    def analyze_errors(
        self,
        predictions: List[str],
        references: List[str],
        task: str = "s2t",
        num_examples: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze prediction errors.
        
        Args:
            predictions: List of predictions
            references: List of references
            task: Task type
            num_examples: Number of error examples to return
            
        Returns:
            Error analysis results
        """
        errors = []
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if pred.strip() != ref.strip():
                # Calculate simple similarity
                pred_tokens = set(pred.lower().split())
                ref_tokens = set(ref.lower().split())
                
                if ref_tokens:
                    overlap = len(pred_tokens & ref_tokens) / len(ref_tokens)
                else:
                    overlap = 0.0
                
                errors.append({
                    'index': i,
                    'prediction': pred,
                    'reference': ref,
                    'token_overlap': overlap
                })
        
        # Sort by token overlap (worst first)
        errors.sort(key=lambda x: x['token_overlap'])
        
        # Categorize errors
        error_categories = {
            'complete_miss': [],  # < 20% overlap
            'partial_match': [],  # 20-80% overlap
            'near_match': []      # > 80% overlap
        }
        
        for error in errors:
            overlap = error['token_overlap']
            if overlap < 0.2:
                error_categories['complete_miss'].append(error)
            elif overlap < 0.8:
                error_categories['partial_match'].append(error)
            else:
                error_categories['near_match'].append(error)
        
        return {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(predictions) if predictions else 0,
            'categories': {
                k: len(v) for k, v in error_categories.items()
            },
            'worst_examples': errors[:num_examples]
        }


def load_model_for_evaluation(model_path: str, device: str = None):
    """
    Load a trained model for evaluation.
    
    Args:
        model_path: Path to saved model
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    
    logger.info(f"Loading model from {model_path}")
    
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    return model, tokenizer


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("CrystalLMEvaluator module loaded successfully.")
    print("\nExample usage:")
    print("  model, tokenizer = load_model_for_evaluation('path/to/model')")
    print("  evaluator = CrystalLMEvaluator(model, tokenizer)")
    print("  results = evaluator.evaluate_from_file('test.json', task='s2t')")

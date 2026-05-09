#!/usr/bin/env python
"""
Evaluate trained CrystalLM model.

This script evaluates a trained model on a test set and computes
various metrics for structure-text translation quality.
"""

import argparse
import json
import os
import sys
import logging
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crystallm.evaluation.metrics import compute_metrics, print_metrics
from crystallm.evaluation.evaluator import CrystalLMEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate CrystalLM model")
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--test_file', type=str, required=True,
                       help='Path to test data JSON')
    parser.add_argument('--task', type=str, choices=['s2t', 't2s'], required=True,
                       help='Task type (s2t or t2s)')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output file for results')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--num_beams', type=int, default=5,
                       help='Number of beams for beam search')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum generation length')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (None for all)')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Log configuration
    logger.info("=" * 50)
    logger.info("CrystalLM Evaluation")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Test file: {args.test_file}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Num beams: {args.num_beams}")
    logger.info("=" * 50)
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()
    
    # Load test data
    logger.info(f"Loading test data from {args.test_file}")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if args.num_samples:
        test_data = test_data[:args.num_samples]
    
    logger.info(f"Evaluating on {len(test_data)} samples")
    
    # Prepare inputs and references
    if args.task == "s2t":
        inputs = [f"translate structure to text: {item['hssr']}" for item in test_data]
        references = [item['description'] for item in test_data]
    else:
        inputs = [f"translate text to structure: {item['description']}" for item in test_data]
        references = [item['hssr'] for item in test_data]
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = []
    
    for i in tqdm(range(0, len(inputs), args.batch_size)):
        batch = inputs[i:i + args.batch_size]
        
        # Tokenize
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            max_length=args.max_length,
            truncation=True,
            padding=True
        ).to(args.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_length=args.max_length,
                num_beams=args.num_beams,
                early_stopping=True
            )
        
        # Decode
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(predictions, references, task=args.task)
    
    # Print results
    print_metrics(metrics, task=args.task)
    
    # Prepare output
    output = {
        'model_path': args.model_path,
        'test_file': args.test_file,
        'task': args.task,
        'num_samples': len(test_data),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'batch_size': args.batch_size,
            'num_beams': args.num_beams,
            'max_length': args.max_length
        },
        'examples': []
    }
    
    # Add sample predictions
    num_examples = min(100, len(predictions))
    for i in range(num_examples):
        output['examples'].append({
            'input': inputs[i],
            'prediction': predictions[i],
            'reference': references[i]
        })
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {args.output_file}")
    
    # Summary
    logger.info("=" * 50)
    logger.info("Evaluation Summary")
    logger.info("=" * 50)
    
    if args.task == "s2t":
        logger.info(f"BLEU-4: {metrics.get('bleu_4', 0):.4f}")
        logger.info(f"ROUGE-L: {metrics.get('rougeL', 0):.4f}")
        logger.info(f"BERTScore F1: {metrics.get('bertscore_f1', 0):.4f}")
    else:
        logger.info(f"Exact Match: {metrics.get('exact_match', 0):.4f}")
        logger.info(f"Validity Rate: {metrics.get('validity_rate', 0):.4f}")
        logger.info(f"Token Accuracy: {metrics.get('token_accuracy', 0):.4f}")
    
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

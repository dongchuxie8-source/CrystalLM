"""
Evaluation metrics for CrystalLM.

This module implements various metrics for evaluating structure-text
translation quality, including BLEU, ROUGE, BERTScore, and
domain-specific metrics for HSSR validation.
"""

from typing import List, Dict, Optional
import numpy as np
from collections import Counter
import re


def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute BLEU scores (1-4).
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Dictionary with BLEU-1 to BLEU-4 scores
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except ImportError:
        print("Warning: nltk not installed. BLEU scores will be 0.")
        return {f'bleu_{n}': 0.0 for n in range(1, 5)}
    
    # Tokenize
    pred_tokens = [pred.split() for pred in predictions]
    ref_tokens = [[ref.split()] for ref in references]
    
    # Smoothing function for short sentences
    smoothing = SmoothingFunction().method1
    
    # Compute BLEU-1 to BLEU-4
    bleu_scores = {}
    for n in range(1, 5):
        weights = tuple([1.0/n] * n + [0.0] * (4-n))
        try:
            score = corpus_bleu(
                ref_tokens, 
                pred_tokens, 
                weights=weights,
                smoothing_function=smoothing
            )
            bleu_scores[f'bleu_{n}'] = score
        except Exception:
            bleu_scores[f'bleu_{n}'] = 0.0
    
    return bleu_scores


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE scores.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("Warning: rouge_score not installed. ROUGE scores will be 0.")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(score[key].fmeasure)
    
    return {key: np.mean(values) for key, values in scores.items()}


def compute_bertscore(
    predictions: List[str], 
    references: List[str],
    lang: str = 'en'
) -> Dict[str, float]:
    """
    Compute BERTScore.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        lang: Language code
        
    Returns:
        Dictionary with BERTScore precision, recall, and F1
    """
    try:
        from bert_score import score as bert_score
    except ImportError:
        print("Warning: bert_score not installed. BERTScore will be 0.")
        return {
            'bertscore_precision': 0.0,
            'bertscore_recall': 0.0,
            'bertscore_f1': 0.0
        }
    
    P, R, F1 = bert_score(predictions, references, lang=lang, verbose=False)
    
    return {
        'bertscore_precision': P.mean().item(),
        'bertscore_recall': R.mean().item(),
        'bertscore_f1': F1.mean().item()
    }


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Compute exact match accuracy.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Exact match accuracy (0-1)
    """
    if not predictions:
        return 0.0
    
    matches = sum(
        1 for pred, ref in zip(predictions, references) 
        if pred.strip() == ref.strip()
    )
    return matches / len(predictions)


def compute_field_accuracy(
    predictions: List[str], 
    references: List[str]
) -> Dict[str, float]:
    """
    Compute field-level accuracy for HSSR.
    
    Evaluates how accurately each field in the HSSR format is predicted.
    
    Args:
        predictions: List of predicted HSSR strings
        references: List of reference HSSR strings
        
    Returns:
        Dictionary with accuracy for each field
    """
    from crystallm.data.hssr_encoder import HSSREncoder
    encoder = HSSREncoder()
    
    fields = ['formula', 'topology', 'space_group', 'pore_diameter', 'density']
    accuracies = {field: [] for field in fields}
    
    for pred, ref in zip(predictions, references):
        try:
            pred_features = encoder.decode(pred)
            ref_features = encoder.decode(ref)
            
            for field in fields:
                if field in ref_features:
                    pred_val = pred_features.get(field)
                    ref_val = ref_features.get(field)
                    
                    if isinstance(ref_val, (int, float)):
                        # Numerical field - check if within 10% tolerance
                        if pred_val is not None and ref_val != 0:
                            relative_error = abs(pred_val - ref_val) / abs(ref_val)
                            accuracies[field].append(relative_error < 0.1)
                        else:
                            accuracies[field].append(pred_val == ref_val)
                    else:
                        # Categorical field - exact match
                        accuracies[field].append(pred_val == ref_val)
        except Exception:
            # If decoding fails, count as incorrect
            for field in fields:
                accuracies[field].append(False)
    
    return {
        f'{field}_acc': np.mean(values) if values else 0.0 
        for field, values in accuracies.items()
    }


def compute_validity_rate(predictions: List[str]) -> float:
    """
    Compute HSSR validity rate.
    
    Checks what percentage of generated HSSR strings are valid.
    
    Args:
        predictions: List of predicted HSSR strings
        
    Returns:
        Validity rate (0-1)
    """
    from crystallm.data.hssr_encoder import HSSREncoder
    encoder = HSSREncoder()
    
    valid_count = 0
    for pred in predictions:
        is_valid, _ = encoder.validate(pred)
        if is_valid:
            valid_count += 1
    
    return valid_count / len(predictions) if predictions else 0.0


def compute_token_accuracy(
    predictions: List[str], 
    references: List[str]
) -> float:
    """
    Compute token-level accuracy.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Token accuracy (0-1)
    """
    total_tokens = 0
    correct_tokens = 0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        # Align tokens
        min_len = min(len(pred_tokens), len(ref_tokens))
        for i in range(min_len):
            if pred_tokens[i] == ref_tokens[i]:
                correct_tokens += 1
        
        total_tokens += len(ref_tokens)
    
    return correct_tokens / total_tokens if total_tokens > 0 else 0.0


def compute_metrics(
    predictions: List[str],
    references: List[str],
    task: str = "s2t"
) -> Dict[str, float]:
    """
    Compute all metrics for evaluation.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        task: Task type ("s2t" for structure-to-text, "t2s" for text-to-structure)
        
    Returns:
        Dictionary of all metric scores
    """
    metrics = {}
    
    # Common metrics
    metrics.update(compute_bleu(predictions, references))
    metrics.update(compute_rouge(predictions, references))
    
    # Task-specific metrics
    if task == "s2t":
        # Structure-to-Text metrics
        metrics.update(compute_bertscore(predictions, references))
    else:
        # Text-to-Structure metrics
        metrics['exact_match'] = compute_exact_match(predictions, references)
        metrics.update(compute_field_accuracy(predictions, references))
        metrics['validity_rate'] = compute_validity_rate(predictions)
        metrics['token_accuracy'] = compute_token_accuracy(predictions, references)
    
    return metrics


def print_metrics(metrics: Dict[str, float], task: str = "s2t"):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        task: Task type
    """
    print("\n" + "=" * 50)
    print(f"Evaluation Results ({task.upper()})")
    print("=" * 50)
    
    # Group metrics
    bleu_metrics = {k: v for k, v in metrics.items() if k.startswith('bleu')}
    rouge_metrics = {k: v for k, v in metrics.items() if k.startswith('rouge')}
    bert_metrics = {k: v for k, v in metrics.items() if k.startswith('bertscore')}
    other_metrics = {k: v for k, v in metrics.items() 
                    if not any(k.startswith(p) for p in ['bleu', 'rouge', 'bertscore'])}
    
    if bleu_metrics:
        print("\nBLEU Scores:")
        for k, v in sorted(bleu_metrics.items()):
            print(f"  {k}: {v:.4f}")
    
    if rouge_metrics:
        print("\nROUGE Scores:")
        for k, v in sorted(rouge_metrics.items()):
            print(f"  {k}: {v:.4f}")
    
    if bert_metrics:
        print("\nBERTScore:")
        for k, v in sorted(bert_metrics.items()):
            print(f"  {k}: {v:.4f}")
    
    if other_metrics:
        print("\nOther Metrics:")
        for k, v in sorted(other_metrics.items()):
            print(f"  {k}: {v:.4f}")
    
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # Example usage
    predictions = [
        "MOF-5 is a zinc-based metal-organic framework with high porosity.",
        "HKUST-1 is a copper-based MOF with paddlewheel secondary building units."
    ]
    
    references = [
        "MOF-5 is a zinc-based metal-organic framework with large pores.",
        "HKUST-1 is a copper MOF featuring paddlewheel clusters."
    ]
    
    # Compute S2T metrics
    metrics = compute_metrics(predictions, references, task="s2t")
    print_metrics(metrics, task="s2t")
    
    # Example T2S evaluation
    hssr_predictions = [
        "[CRYSTAL] [META] MOF MOF-5 [COMP] Zn4O(C8H4O4)3 [/CRYSTAL]",
        "[CRYSTAL] [META] MOF HKUST-1 [COMP] Cu3(C9H3O6)2 [/CRYSTAL]"
    ]
    
    hssr_references = [
        "[CRYSTAL] [META] MOF MOF-5 [COMP] Zn4O(C8H4O4)3 [/CRYSTAL]",
        "[CRYSTAL] [META] MOF HKUST-1 [COMP] Cu3(C9H3O6)2 [/CRYSTAL]"
    ]
    
    t2s_metrics = compute_metrics(hssr_predictions, hssr_references, task="t2s")
    print_metrics(t2s_metrics, task="t2s")

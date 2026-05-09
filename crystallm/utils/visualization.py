"""
Visualization utilities for CrystalLM.

This module provides functions for visualizing training progress,
evaluation results, and model analysis.
"""

import os
from typing import List, Dict, Optional, Any
import json


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    output_path: Optional[str] = None,
    title: str = "Training Curves"
) -> Any:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        output_path: Path to save the plot (optional)
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed. Cannot create plots.")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Mark best validation loss
    best_epoch = val_losses.index(min(val_losses)) + 1
    best_loss = min(val_losses)
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    ax.annotate(
        f'Best: {best_loss:.4f}',
        xy=(best_epoch, best_loss),
        xytext=(best_epoch + 0.5, best_loss + 0.05),
        fontsize=10
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    return fig


def plot_metrics(
    metrics: Dict[str, float],
    output_path: Optional[str] = None,
    title: str = "Evaluation Metrics"
) -> Any:
    """
    Plot evaluation metrics as a bar chart.
    
    Args:
        metrics: Dictionary of metric names and values
        output_path: Path to save the plot (optional)
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not installed. Cannot create plots.")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort metrics by name
    sorted_metrics = dict(sorted(metrics.items()))
    names = list(sorted_metrics.keys())
    values = list(sorted_metrics.values())
    
    # Create bar chart
    x = np.arange(len(names))
    bars = ax.bar(x, values, color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f'{value:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=9
        )
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    return fig


def plot_comparison(
    results: Dict[str, Dict[str, float]],
    metric_names: List[str],
    output_path: Optional[str] = None,
    title: str = "Model Comparison"
) -> Any:
    """
    Plot comparison of multiple models.
    
    Args:
        results: Dictionary mapping model names to their metrics
        metric_names: List of metrics to compare
        output_path: Path to save the plot (optional)
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not installed. Cannot create plots.")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_names = list(results.keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(model_names)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    
    for i, (model_name, metrics) in enumerate(results.items()):
        values = [metrics.get(m, 0) for m in metric_names]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i])
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    return fig


def plot_attention_heatmap(
    attention_weights,
    input_tokens: List[str],
    output_tokens: List[str],
    output_path: Optional[str] = None,
    title: str = "Attention Weights"
) -> Any:
    """
    Plot attention weights as a heatmap.
    
    Args:
        attention_weights: 2D array of attention weights
        input_tokens: List of input tokens
        output_tokens: List of output tokens
        output_path: Path to save the plot (optional)
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not installed. Cannot create plots.")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Attention Weight', rotation=-90, va="bottom")
    
    # Set ticks
    ax.set_xticks(np.arange(len(input_tokens)))
    ax.set_yticks(np.arange(len(output_tokens)))
    ax.set_xticklabels(input_tokens, rotation=90)
    ax.set_yticklabels(output_tokens)
    
    ax.set_xlabel('Input Tokens', fontsize=12)
    ax.set_ylabel('Output Tokens', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    return fig


def create_results_table(
    results: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None
) -> str:
    """
    Create a formatted results table.
    
    Args:
        results: Dictionary mapping model names to their metrics
        output_path: Path to save the table (optional)
        
    Returns:
        Formatted table string
    """
    # Get all metric names
    all_metrics = set()
    for metrics in results.values():
        all_metrics.update(metrics.keys())
    metric_names = sorted(all_metrics)
    
    # Create header
    header = "| Model | " + " | ".join(metric_names) + " |"
    separator = "|" + "|".join(["---"] * (len(metric_names) + 1)) + "|"
    
    # Create rows
    rows = []
    for model_name, metrics in results.items():
        values = [f"{metrics.get(m, 0):.4f}" for m in metric_names]
        row = f"| {model_name} | " + " | ".join(values) + " |"
        rows.append(row)
    
    # Combine
    table = "\n".join([header, separator] + rows)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(table)
        print(f"Table saved to {output_path}")
    
    return table


def visualize_hssr_structure(
    hssr_string: str,
    output_path: Optional[str] = None
) -> str:
    """
    Create a visual representation of HSSR structure.
    
    Args:
        hssr_string: HSSR formatted string
        output_path: Path to save visualization (optional)
        
    Returns:
        ASCII art representation
    """
    lines = hssr_string.strip().split('\n')
    
    # Create tree-like visualization
    output = []
    output.append("HSSR Structure Visualization")
    output.append("=" * 40)
    
    indent_level = 0
    for line in lines:
        stripped = line.strip()
        
        # Adjust indent based on tags
        if stripped.startswith('[/'):
            indent_level = max(0, indent_level - 1)
        
        # Add line with proper indentation
        prefix = "  " * indent_level
        if stripped.startswith('[') and not stripped.startswith('[/'):
            output.append(f"{prefix}+-- {stripped}")
            if not stripped.endswith(']') or stripped.count('[') > stripped.count(']'):
                indent_level += 1
        else:
            output.append(f"{prefix}    {stripped}")
    
    output.append("=" * 40)
    
    result = "\n".join(output)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(result)
        print(f"Visualization saved to {output_path}")
    
    return result


if __name__ == "__main__":
    # Example usage
    
    # Training curves
    train_losses = [0.8, 0.6, 0.5, 0.45, 0.42, 0.40, 0.38, 0.37]
    val_losses = [0.75, 0.55, 0.48, 0.44, 0.43, 0.42, 0.41, 0.40]
    
    print("Plotting training curves...")
    plot_training_curves(train_losses, val_losses, title="CrystalLM Training")
    
    # Metrics
    metrics = {
        'bleu_1': 0.65,
        'bleu_2': 0.52,
        'bleu_3': 0.42,
        'bleu_4': 0.35,
        'rouge1': 0.58,
        'rouge2': 0.45,
        'rougeL': 0.52
    }
    
    print("\nPlotting metrics...")
    plot_metrics(metrics, title="S2T Evaluation Metrics")
    
    # Results table
    results = {
        'T5-base': {'bleu_4': 0.32, 'rouge1': 0.48, 'bertscore_f1': 0.72},
        'T5-large': {'bleu_4': 0.40, 'rouge1': 0.56, 'bertscore_f1': 0.78},
        'T5-3B': {'bleu_4': 0.45, 'rouge1': 0.60, 'bertscore_f1': 0.82}
    }
    
    print("\nResults Table:")
    print(create_results_table(results))
    
    # HSSR visualization
    hssr = """[CRYSTAL]
  [META] MOF MOF-5
  [COMP] Zn4O(C8H4O4)3
  [LATT] a=25.832 b=25.832 c=25.832
  [SYMM] Fm-3m 225 cubic
  [TOPO] pcu
[/CRYSTAL]"""
    
    print("\nHSSR Visualization:")
    print(visualize_hssr_structure(hssr))

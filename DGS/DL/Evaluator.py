"""Evaluate Metrics provided in DGS."""

import os
import logging
import itertools
from typing import Dict, List, Tuple, Union, Optional, Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc, roc_curve, precision_recall_curve, 
    accuracy_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from scipy.stats import pearsonr, spearmanr, kendalltau

logger = logging.getLogger("dgs.evaluator")

def onehot_encode(label: np.ndarray) -> np.ndarray:
    """Convert integer labels to one-hot encoded format."""
    from sklearn.preprocessing import label_binarize
    return label_binarize(label, classes=range(np.max(label)+1))

def metrics_to_df(metrics: Dict) -> pd.DataFrame:
    """Convert metrics dictionary to DataFrame."""
    return pd.DataFrame(metrics).T

def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    return_dict: bool = False
) -> Union[pd.DataFrame, Dict]:
    """Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        return_dict: Whether to return dictionary instead of DataFrame
        
    Returns:
        DataFrame or Dictionary containing classification metrics
    """
    # infer if this is a multiclass problem
    multi_class = len(y_pred.shape) > 1

    if multi_class and y_true.ndim == 1:
        y_true = onehot_encode(y_true)

    if not multi_class:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)

        y_pred = (y_pred > threshold).astype(int)

        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        metrics = {
            'auroc': roc_auc,
            'auprc': pr_auc,
            'f1': f1,
            'accuracy': accuracy
        }

        return metrics if return_dict else metrics_to_df({'task': metrics})
    
    else:
        # Calculate metrics
        metrics = {}
        
        if len(y_true.shape) == 1:
            y_true = onehot_encode(y_true)
        
        n_classes = y_true.shape[1]
        
        # Per-class metrics
        for i in range(n_classes):
            metrics[f"task_{i}"] = calculate_classification_metrics(y_true[:, i], y_pred[:, i], return_dict=True)

        return metrics if return_dict else metrics_to_df(metrics)


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    return_dict: bool = False
) -> Union[pd.DataFrame, Dict]:
    """Calculate regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        return_dict: Whether to return dictionary instead of DataFrame
        
    Returns:
        DataFrame or Dictionary containing regression metrics
    """
    # infer if this is a multiclass problem
    multi_class = len(y_pred.shape) > 1

    if not multi_class:
        # Ensure 1D arrays
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Correlation metrics
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
        kendall_tau, kendall_p = kendalltau(y_true, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'kendall_tau': kendall_tau,
            'kendall_p': kendall_p
        }

        return metrics if return_dict else metrics_to_df({'task': metrics})
    
    else:
        # Calculate metrics
        metrics = {}
        
        if len(y_true.shape) == 1:
            y_true = onehot_encode(y_true)

        n_classes = y_true.shape[1]

        for i in range(n_classes):
            metrics[f"task_{i}"] = calculate_regression_metrics(y_true[:, i], y_pred[:, i], return_dict=True)

        return metrics if return_dict else metrics_to_df(metrics)


def calculate_sequence_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    mask: Optional[np.ndarray] = None,
    return_dict: bool = False
) -> Union[pd.DataFrame, Dict]:
    """Calculate sequence classification metrics.
    
    Args:
        y_true: Ground truth sequences, shape (n_samples, seq_len, n_classes)
        y_pred: Predicted probabilities, shape (n_samples, seq_len, n_classes)
        mask: Optional mask for valid positions, shape (n_samples, seq_len)
        
    Returns:
        Dictionary containing sequence-level metrics
    """
    n_samples, seq_len, n_classes = y_true.shape

    # Calculate metrics only on masked positions
    if mask is None:
        mask = np.ones((n_samples, seq_len), dtype=bool)

    # Per-sample metrics
    metrics_per_sample = {}
    for sample_idx in range(n_samples):
        seq_mask = mask[sample_idx]
            
        sample_true = y_true[sample_idx][seq_mask, :]
        sample_pred = y_pred[sample_idx][seq_mask, :]
                
        # classification metrics
        sample_metrics = calculate_classification_metrics(sample_true, sample_pred, threshold=threshold, return_dict=True)
        for metric_name in sample_metrics.keys():
            metrics_per_sample[f'sample_{sample_idx}_{metric_name}'] = sample_metrics[metric_name]

    # stack-all-samples-positions-metrics
    mask = mask.ravel()
    y_true = y_true.reshape(-1, n_classes)[mask]
    y_pred = y_pred.reshape(-1, n_classes)[mask]
    metrics = calculate_classification_metrics(y_true, y_pred, threshold=threshold, return_dict=True)

    if not return_dict:
        metrics = pd.DataFrame(metrics).T
        metrics_per_sample = pd.DataFrame(metrics_per_sample).T
    
    return metrics, metrics_per_sample


def calculate_sequence_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
    return_dict: bool = False
) -> Union[pd.DataFrame, Dict]:
    """Calculate sequence regression metrics.
    
    Args:
        y_true: Ground truth sequences, shape (n_samples, seq_len, n_classes)
        y_pred: Predicted values, shape (n_samples, seq_len, n_classes)
        mask: Optional mask for valid positions, shape (n_samples, seq_len)
        
    Returns:
        Dictionary containing sequence-level metrics
    """
    n_samples, seq_len, n_classes = y_true.shape

    # Calculate metrics only on masked positions
    if mask is None:
        mask = np.ones((n_samples, seq_len), dtype=bool)

    # Per-sample metrics
    metrics_per_sample = {}
    for sample_idx in range(n_samples):
        seq_mask = mask[sample_idx]
            
        sample_true = y_true[sample_idx][seq_mask, :]
        sample_pred = y_pred[sample_idx][seq_mask, :]
                
        # classification metrics
        sample_metrics = calculate_regression_metrics(sample_true, sample_pred, return_dict=True)
        for metric_name in sample_metrics.keys():
            metrics_per_sample[f'sample_{sample_idx}_{metric_name}'] = sample_metrics[metric_name]

    # stack-all-samples-positions-metrics
    mask = mask.ravel()
    y_true = y_true.reshape(-1, n_classes)[mask]
    y_pred = y_pred.reshape(-1, n_classes)[mask]
    metrics = calculate_regression_metrics(y_true, y_pred, return_dict=True)

    if not return_dict:
        metrics = pd.DataFrame(metrics).T
        metrics_per_sample = pd.DataFrame(metrics_per_sample).T
    
    return metrics, metrics_per_sample


def show_auc_curve(metrics,
                    fig_size=(10,8),
                    save=True,
                    output_dir='Figures',
                    output_fname='roc_curves.pdf',
                    fig_title="Feature ROC curves",
                    lw=1, label=True,
                    dpi=500):
    import matplotlib
    backend = matplotlib.get_backend()
    if "inline" not in backend:
        matplotlib.use("PDF")
    import matplotlib.pyplot as plt
    plt.figure()
    # n_classes
    n_classes = len(metrics)
    # Plot all ROC curves
    plt.figure(figsize=fig_size)
    colors = ["grey"]
    labels = [f"task_{i}" for i in range(n_classes)]
    for i, color in zip(range(n_classes), itertools.cycle(colors)):
        plt.plot(metrics[f"task_{i}"].get("fpr", [0]), 
            metrics[f"task_{i}"].get("tpr", [0]), 
            color=color, lw=lw,
            label=labels[i] if label else None
            )
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(fig_title)
    plt.legend(loc="lower right")
    if save:
        plt.savefig(os.path.join(output_dir, output_fname),
                    format="pdf", dpi=dpi)
    else:
        plt.show()
    plt.close()


def show_pr_curve(metrics,
                    fig_size=(10,8),
                    save=True,
                    output_dir='Figures',
                    output_fname='pr_curves.pdf',
                    fig_title="Feature PR curves",
                    lw=1, label=True,
                    dpi=500):
    import matplotlib
    backend = matplotlib.get_backend()
    if "inline" not in backend:
        matplotlib.use("PDF")
    import matplotlib.pyplot as plt
    plt.figure()
    # n_classes
    n_classes = len(metrics)
    # Plot all ROC curves
    plt.figure(figsize=fig_size)
    colors = ["grey"]
    labels = [f"task_{i}" for i in range(n_classes)]
    for i, color in zip(range(n_classes), itertools.cycle(colors)):
        plt.plot(metrics[f"task_{i}"].get("recall", [0]), 
            metrics[f"task_{i}"].get("precision", [0]), 
            color=color, lw=lw,
            label=labels[i] if label else None
            )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(fig_title)
    plt.legend(loc="lower right")
    if save:
        plt.savefig(os.path.join(output_dir, output_fname),
                    format="pdf", dpi=dpi)
    else:
        plt.show()
    plt.close()




"""
Model Evaluation and Metrics Module

This module provides comprehensive evaluation tools for assessing model performance
on genomic sequence analysis tasks. It supports both classification and regression
metrics for various prediction scenarios.

Key Components:
1. Classification Metrics:
   - Binary and multi-class classification
   - ROC curves and AUC calculation
   - Precision-recall analysis
   - F1 score and accuracy metrics

2. Regression Metrics:
   - Mean squared error (MSE)
   - Root mean squared error (RMSE)
   - Mean absolute error (MAE)
   - R-squared and correlation coefficients

3. Sequence-Level Metrics:
   - Position-wise performance analysis
   - Sequence-averaged metrics
   - Masked evaluation support
   - Per-sample statistics

4. Visualization Tools:
   - ROC curve plotting
   - Precision-recall curve visualization
   - Performance comparison plots
   - Metric distribution analysis

The module is designed to handle various prediction formats:
- Single-task and multi-task models
- Binary and multi-class classification
- Continuous value regression
- Sequence-to-sequence prediction
"""

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
    """
    Convert integer labels to one-hot encoded format.

    Args:
        label (np.ndarray): Integer labels array

    Returns:
        np.ndarray: One-hot encoded labels
            Shape: (n_samples, n_classes)

    Note:
        Automatically determines number of classes from input data.
    """
    from sklearn.preprocessing import label_binarize
    return label_binarize(label, classes=range(np.max(label)+1))

def metrics_to_df(metrics: Dict) -> pd.DataFrame:
    """
    Convert metrics dictionary to DataFrame format.

    Args:
        metrics (Dict): Dictionary of metric names and values

    Returns:
        pd.DataFrame: Metrics in tabular format
            Index: Metric names
            Values: Metric scores
    """
    return pd.DataFrame(metrics).T

def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    return_dict: bool = False
) -> Union[pd.DataFrame, Dict]:
    """
    Calculate comprehensive classification performance metrics.

    This function computes various metrics for both binary and
    multi-class classification tasks:
    - ROC AUC score
    - Precision-Recall AUC
    - F1 score
    - Accuracy
    - Per-class metrics for multi-class problems

    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Model predictions (probabilities)
        threshold (float): Classification threshold for binary tasks
        return_dict (bool): Whether to return dictionary instead of DataFrame

    Returns:
        Union[pd.DataFrame, Dict]: Classification metrics
            For binary classification:
                - auroc: Area under ROC curve
                - auprc: Area under precision-recall curve
                - f1: F1 score
                - accuracy: Classification accuracy
            For multi-class:
                - Per-class metrics in separate rows/keys

    Note:
        Automatically handles both binary and multi-class cases.
        For multi-class, computes metrics for each class separately.
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
    """
    Calculate comprehensive regression performance metrics.

    This function computes various regression metrics:
    - Mean squared error (MSE)
    - Root mean squared error (RMSE)
    - Mean absolute error (MAE)
    - R-squared score
    - Correlation coefficients (Pearson, Spearman, Kendall)

    Args:
        y_true (np.ndarray): Ground truth values
        y_pred (np.ndarray): Model predictions
        return_dict (bool): Whether to return dictionary instead of DataFrame

    Returns:
        Union[pd.DataFrame, Dict]: Regression metrics
            - mse: Mean squared error
            - rmse: Root mean squared error
            - mae: Mean absolute error
            - r2: R-squared score
            - pearson_r: Pearson correlation coefficient
            - pearson_p: Pearson p-value
            - spearman_r: Spearman correlation coefficient
            - spearman_p: Spearman p-value
            - kendall_tau: Kendall's tau
            - kendall_p: Kendall's p-value

    Note:
        Handles both single-task and multi-task regression.
        For multi-task, computes metrics for each task separately.
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
    """
    Calculate classification metrics for sequence prediction tasks.

    This function handles sequence-to-sequence classification:
    - Computes metrics at each sequence position
    - Supports masked evaluation
    - Provides both per-sample and aggregate metrics
    - Handles multi-class sequence labeling

    Args:
        y_true (np.ndarray): Ground truth sequences (n_samples, seq_len, n_classes)
        y_pred (np.ndarray): Predicted probabilities (n_samples, seq_len, n_classes)
        threshold (float): Classification threshold
        mask (np.ndarray, optional): Mask for valid positions
        return_dict (bool): Whether to return dictionary instead of DataFrame

    Returns:
        Union[pd.DataFrame, Dict]: Two sets of metrics:
            1. Aggregate metrics across all positions
            2. Per-sample metrics for detailed analysis

    Note:
        Mask can be used to exclude padding or uncertain positions
        from evaluation.
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
    """
    Calculate regression metrics for sequence prediction tasks.

    This function handles sequence-to-sequence regression:
    - Computes metrics at each sequence position
    - Supports masked evaluation
    - Provides both per-sample and aggregate metrics
    - Handles multi-task sequence prediction

    Args:
        y_true (np.ndarray): Ground truth sequences (n_samples, seq_len, n_tasks)
        y_pred (np.ndarray): Predicted values (n_samples, seq_len, n_tasks)
        mask (np.ndarray, optional): Mask for valid positions
        return_dict (bool): Whether to return dictionary instead of DataFrame

    Returns:
        Union[pd.DataFrame, Dict]: Two sets of metrics:
            1. Aggregate metrics across all positions
            2. Per-sample metrics for detailed analysis

    Note:
        Mask can be used to exclude padding or uncertain positions
        from evaluation.
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
    """
    Plot ROC curves for classification results.

    This function visualizes ROC curves:
    - Supports multiple tasks/classes
    - Customizable plot appearance
    - Optional saving to file
    - Automatic figure management

    Args:
        metrics (Dict): Dictionary containing ROC curve data
        fig_size (tuple): Figure size (width, height)
        save (bool): Whether to save plot to file
        output_dir (str): Directory for saving plots
        output_fname (str): Filename for saved plot
        fig_title (str): Plot title
        lw (float): Line width
        label (bool): Whether to show labels
        dpi (int): Resolution for saved plot

    Note:
        Creates a new figure for each call and closes it
        after saving to prevent memory leaks.
    """
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
    """
    Plot precision-recall curves for classification results.

    This function visualizes precision-recall curves:
    - Supports multiple tasks/classes
    - Customizable plot appearance
    - Optional saving to file
    - Automatic figure management

    Args:
        metrics (Dict): Dictionary containing PR curve data
        fig_size (tuple): Figure size (width, height)
        save (bool): Whether to save plot to file
        output_dir (str): Directory for saving plots
        output_fname (str): Filename for saved plot
        fig_title (str): Plot title
        lw (float): Line width
        label (bool): Whether to show labels
        dpi (int): Resolution for saved plot

    Note:
        Creates a new figure for each call and closes it
        after saving to prevent memory leaks.
    """
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




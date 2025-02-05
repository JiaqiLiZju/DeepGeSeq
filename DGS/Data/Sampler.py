"""Sampling and splitting utilities for genomic datasets.

This module provides:
1. Dataset splitting strategies (random, chromosome-based)
2. DataLoader creation utilities
"""

import numpy as np
from typing import Union, List, Tuple, Optional

from .Interval import Interval

def random_split(
    intervals: Interval,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_state: Optional[int] = None
) -> Union[Tuple[Interval, Interval], Tuple[Interval, Interval, Interval]]:
    """Split dataset randomly.
    
    Args:
        dataset: Dataset to split
        test_size: Proportion of data for test set
        val_size: Optional proportion for validation set
        stratify: Whether to use stratified splitting (for labeled data)
        random_state: Random seed
        
    Returns:
        Split datasets (train, test) or (train, val, test)
    """
    # Get intervals for splitting
    intervals = intervals.copy()
    
    # Generate random masks for splitting
    rng = np.random.RandomState(random_state)
    indices = np.arange(len(intervals.data))
    
    # Create test split
    test_size_int = int(len(indices) * test_size)
    test_indices = rng.choice(indices, size=test_size_int, replace=False)
    test_intervals = intervals.__class__(intervals.data.iloc[test_indices])
    
    if val_size is not None:
        # Create validation split from remaining data
        remaining_indices = np.setdiff1d(indices, test_indices)
        val_size_int = int(len(remaining_indices) * val_size / (1 - test_size))
        val_indices = rng.choice(remaining_indices, size=val_size_int, replace=False)
        train_indices = np.setdiff1d(remaining_indices, val_indices)
        
        # Create interval subsets
        val_intervals = intervals.__class__(intervals.data.iloc[val_indices])
        train_intervals = intervals.__class__(intervals.data.iloc[train_indices])
        
        return (train_intervals, val_intervals, test_intervals)
    
    # Create train split
    train_indices = np.setdiff1d(indices, test_indices)
    train_intervals = intervals.__class__(intervals.data.iloc[train_indices])
    
    return (train_intervals, test_intervals)

def chromosome_split(
    intervals: Interval,
    test_chroms: List[str],
    val_chroms: Optional[List[str]] = None
) -> Union[Tuple[Interval, Interval], Tuple[Interval, Interval, Interval]]:
    """Split dataset by chromosomes.
    
    Args:
        dataset: Dataset to split
        test_chroms: Chromosomes for test set
        val_chroms: Optional chromosomes for validation set
        
    Returns:
        Split datasets by chromosome
    """
    intervals = intervals.copy()
    
    # Create test intervals
    test_mask = intervals.data['chrom'].isin(test_chroms)
    test_intervals = intervals.__class__(intervals.data[test_mask])
    
    if val_chroms:
        # Create validation and train intervals
        val_mask = intervals.data['chrom'].isin(val_chroms)
        train_mask = ~(test_mask | val_mask)
        
        val_intervals = intervals.__class__(intervals.data[val_mask])
        train_intervals = intervals.__class__(intervals.data[train_mask])
        
        return (train_intervals, val_intervals, test_intervals)
    
    # Create train intervals
    train_intervals = intervals.__class__(intervals.data[~test_mask])
    
    return (train_intervals, test_intervals)


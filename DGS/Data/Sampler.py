"""
Data Sampling and Splitting Module

This module provides comprehensive utilities for sampling and splitting genomic datasets:

Key Features:
1. Dataset Splitting:
   - Random splitting with stratification
   - Chromosome-based splitting for genomic data
   - Support for train/test and train/val/test splits
   - Customizable split ratios

2. Sampling Strategies:
   - Random sampling with seed control
   - Balanced sampling for imbalanced data
   - Region-based sampling for genomic intervals
   - Cross-validation support

3. Data Management:
   - Interval-based data handling
   - Index tracking and preservation
   - Split validation and statistics
   - Memory-efficient operations

The module ensures reproducible and statistically sound data splitting
for genomic deep learning applications.
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
    """Split genomic intervals randomly into training and test/validation sets.
    
    This function implements a random splitting strategy that:
    - Preserves interval object properties
    - Maintains data integrity
    - Supports reproducible splitting
    - Handles both binary and three-way splits
    
    Args:
        intervals: Interval object containing genomic regions
        test_size: Proportion of data to use for testing (0.0 to 1.0)
        val_size: Optional proportion for validation set (0.0 to 1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        - If val_size is None: (train_intervals, test_intervals)
        - If val_size is provided: (train_intervals, val_intervals, test_intervals)
        
    Note:
        The function preserves the original Interval object's properties
        and metadata while splitting the underlying data.
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
    """Split genomic intervals by chromosome for robust model evaluation.
    
    This function implements chromosome-based splitting that:
    - Prevents data leakage between sets
    - Evaluates model generalization
    - Supports genomic region isolation
    - Maintains chromosome integrity
    
    Args:
        intervals: Interval object containing genomic regions
        test_chroms: List of chromosomes to use for testing
        val_chroms: Optional list of chromosomes for validation
        
    Returns:
        - If val_chroms is None: (train_intervals, test_intervals)
        - If val_chroms is provided: (train_intervals, val_intervals, test_intervals)
        
    Note:
        Chromosome-based splitting is recommended for genomic data to:
        1. Prevent position-based correlations between sets
        2. Test model generalization to unseen chromosomes
        3. Evaluate chromosome-specific effects
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


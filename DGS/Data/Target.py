"""
Genomic Target Data Management Module

This module provides a comprehensive framework for handling genomic target data:

Key Components:
1. Target Data Processing:
   - Support for BED and BigWig formats
   - Multi-task data management
   - Flexible data encoding
   - Efficient storage

2. Data Analysis Features:
   - Statistical calculations
   - Distribution analysis
   - Quality metrics
   - Validation tools

3. Integration Capabilities:
   - Seamless BED/BigWig reading
   - Interval-based operations
   - Task configuration management
   - Format conversion utilities

The module is designed for:
- Efficient handling of large genomic datasets
- Support for various genomic data formats
- Flexible task configuration
- Robust data validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging

from ..IO import BedReader, BigWigReader
from .Interval import Interval, find_overlaps

logger = logging.getLogger("dgs.target")

class Target:
    """Class for managing genomic target data and annotations.
    
    This class provides a comprehensive framework for:
    - Loading and processing genomic annotations
    - Managing multiple prediction tasks
    - Computing dataset statistics
    - Validating data quality
    
    Features:
    - Support for BED and BigWig formats
    - Multi-task data management
    - Flexible data encoding
    - Statistical analysis
    
    Attributes:
        intervals: Base genomic intervals
        tasks: Task configurations
        task_info: Task statistics
        data: Processed target data
        
    Example:
        >>> intervals = pd.read_csv("regions.bed", sep='\t')
        >>> tasks = [
        ...     {
        ...         'file_path': 'peaks.bed',
        ...         'file_type': 'bed',
        ...         'task_type': 'binary',
        ...         'task_name': 'binding'
        ...     }
        ... ]
        >>> target = Target(intervals, tasks)
        >>> labels = target.get_labels('binding')
    """
    
    def __init__(
        self,
        intervals: Union[pd.DataFrame, str, Path],
        tasks: List[Dict[str, Union[str, float, int]]],
        chrom_col: str = 'chrom',
        start_col: str = 'start',
        end_col: str = 'end'
    ):
        """Initialize Target object.
        
        Args:
            intervals: Input intervals (DataFrame or file path)
            tasks: List of task configurations
                Each task should have:
                - file_path: Path to data file
                - file_type: Type of file ('bed' or 'bigwig')
                - task_type: Type of task ('binary', 'regression')
                - task_name: Name of the task
                Additional parameters for specific file types:
                - For BED: target_column
                - For BigWig: bin_size, aggfunc, threshold
            chrom_col: Name of chromosome column
            start_col: Name of start position column
            end_col: Name of end position column
            
        Note:
            The class automatically validates input data and
            computes relevant statistics for each task.
        """
        # Initialize base intervals
        self.intervals = Interval(intervals, chrom_col, start_col, end_col)
        self.chrom_col = chrom_col
        self.start_col = start_col
        self.end_col = end_col
        
        # Initialize task data
        self.tasks = {}  # Store task configurations
        self.task_info = {}  # Store task statistics
        self.data = {}  # Store NumPy arrays
        self._index = self.intervals.data.index
        
        # Load each task
        for task in tasks:
            task_name = task['task_name']
            self.tasks[task_name] = task  # Store task configuration
            self._load_task(task)
            
    def _load_task(self, task: Dict[str, Union[str, float, int]]) -> None:
        """Load and process data for a single task.
        
        This method:
        1. Validates task configuration
        2. Loads data from file
        3. Processes according to task type
        4. Computes task statistics
        
        Args:
            task: Task configuration dictionary
            
        Note:
            Automatically handles different file formats and
            task types appropriately.
        """
        task_name = task['task_name']
        file_type = task['file_type'].lower()
        
        # Store task information
        self.task_info[task_name] = task
        
        # Load data based on file type
        if file_type == 'bed':
            self._load_from_bed(task)
        elif file_type == 'bigwig':
            self._load_from_bigwig(task)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    def _load_from_bed(self, task: Dict[str, Union[str, float, int]]) -> None:
        """Load and process data from BED file.
        
        This method:
        1. Reads BED file
        2. Finds overlaps with base intervals
        3. Creates label array
        4. Computes statistics
        
        Args:
            task: Task configuration dictionary
            
        Note:
            Handles both binary and categorical labels from
            BED files.
        """
        task_name = task['task_name']
        target_col = task.get('target_column', 'name')
        
        # Read BED file using BedReader
        reader = BedReader(task['file_path'])
        target_data = reader.read()
        
        # Find overlaps with base intervals
        overlaps = find_overlaps(
            self.intervals.data,
            target_data,
            self.chrom_col,
            self.start_col,
            self.end_col
        )
        
        # Create label array
        labels = np.zeros(len(self._index), dtype=np.int8)
        if len(overlaps) > 0:
            if 'query_index' not in overlaps.columns:
                overlaps['query_index'] = overlaps.index
            
            # Get indices in original order
            positive_indices = self._index.get_indexer(overlaps['query_index'].unique())
            
            # Use target column values if available
            if target_col in overlaps.columns:
                labels[positive_indices] = overlaps.groupby('query_index')[target_col].first().values
            else:
                labels[positive_indices] = 1
            
        # Store labels and statistics
        self.data[task_name] = labels
        self.task_info[task_name].update({
            'positive_count': (labels > 0).sum(),
            'negative_count': (labels == 0).sum(),
            'positive_ratio': (labels > 0).mean()
        })
        
    def _load_from_bigwig(self, task: Dict[str, Union[str, float, int]]) -> None:
        """Load and process data from BigWig file.
        
        This method:
        1. Reads signal values
        2. Processes according to parameters
        3. Optionally converts to binary
        4. Computes statistics
        
        Args:
            task: Task configuration dictionary
            
        Note:
            The signal values are stored as 2D array (intervals × bins):
            - bin_size=1: Each row contains base-level signals
            - bin_size>1: Each row contains binned signals
        """
        task_name = task['task_name']
        bin_size = task.get('bin_size', None)
        aggfunc = task.get('aggfunc', 'mean')
        threshold = task.get('threshold', None)
        
        # Read signal values using BigWigReader
        reader = BigWigReader(task['file_path'])
        values = reader.read(
            self.intervals.data,
            bin_size=bin_size,
            aggfunc=aggfunc
        )
        
        # Convert values to float32
        values = values.astype(np.float32)
        
        # Convert to binary labels if threshold is provided
        if threshold is not None:
            values = (values >= threshold).astype(np.int8)
        
        # Store values and statistics
        self.data[task_name] = values
        
        # Calculate basic statistics
        stats = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'bin_size': bin_size,
            'aggfunc': aggfunc,
            'shape': values.shape
        }
        
        # Add binary classification stats if threshold is provided
        if threshold is not None:
            binary = (values >= threshold).astype(np.int8)
            stats.update({
                'threshold': threshold,
                'positive_count': int(binary.sum()),
                'negative_count': int((binary == 0).sum()),
                'positive_ratio': float(binary.mean())
            })
        
        self.task_info[task_name] = stats
        
    def get_labels(
        self,
        task_name: Optional[str] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Get labels for specified task(s).
        
        Args:
            task_name: Name of task (None for all tasks)
            
        Returns:
            - If task_name is provided: numpy array (1D or 2D)
            - If task_name is None: Dictionary of numpy arrays
            
        Note:
            For multiple tasks, returns a dictionary with task
            names as keys and label arrays as values.
        """
        if task_name is not None:
            if task_name not in self.data:
                raise ValueError(f"Unknown task: {task_name}")
            return self.data[task_name]
        else:
            # Return all tasks as a concatenated array with tasks as last dimension
            return np.stack(list(self.data.values()), axis=-1)
        
    def get_stats(
        self,
        task_name: Optional[str] = None
    ) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Get statistics for specified task(s).
        
        Args:
            task_name: Name of task (None for all tasks)
            
        Returns:
            - If task_name is provided: Dictionary of statistics
            - If task_name is None: Dictionary of dictionaries
            
        Note:
            Statistics include distribution metrics, counts,
            and task-specific information.
        """
        if task_name is not None:
            if task_name not in self.task_info:
                raise ValueError(f"Unknown task: {task_name}")
            return self.task_info[task_name]
            
        return self.task_info
        
    def get_intervals(self) -> pd.DataFrame:
        """Get base intervals used for target data.
        
        Returns:
            DataFrame containing interval information
        """
        return self.intervals.data
        
    def get_task_info(self) -> pd.DataFrame:
        """Get comprehensive task information.
        
        Returns:
            DataFrame containing task information with:
            - Tasks as rows
            - Properties as columns
            - Configuration and statistics combined
            
        Note:
            Useful for analyzing task characteristics and
            data distributions.
        """
        # Convert task info dictionary to DataFrame
        task_df = pd.DataFrame.from_dict(self.task_info, orient='index')
        
        # Add task configuration info that might not be in task_info
        for task_name, task_config in self.tasks.items():
            for key, value in task_config.items():
                if key not in task_df.columns:
                    task_df.loc[task_name, key] = value
        
        return task_df
        

# Helper functions
def get_class_distribution(
    labels: Union[np.ndarray, pd.Series],
    normalize: bool = True
) -> pd.Series:
    """Calculate class distribution in label data.
    
    Args:
        labels: Input label array or series
        normalize: Whether to normalize counts
        
    Returns:
        Series with class distribution:
        - Index: Unique classes
        - Values: Counts or proportions
        
    Note:
        Useful for analyzing class imbalance and
        distribution characteristics.
    """
    if isinstance(labels, pd.Series):
        labels = labels.values
        
    unique, counts = np.unique(labels, return_counts=True)
    dist = pd.Series(counts, index=unique)
    
    if normalize:
        dist = dist / len(labels)
        
    return dist.sort_index()
    
def get_imbalance_metrics(labels: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """Calculate comprehensive class imbalance metrics.
    
    This function computes:
    - Imbalance ratio
    - Shannon entropy
    - Gini coefficient
    
    Args:
        labels: Input label array or series
        
    Returns:
        Dictionary of imbalance metrics:
        - imbalance_ratio: Ratio of most to least common
        - entropy: Shannon entropy of distribution
        - gini: Gini coefficient of distribution
        
    Note:
        Higher entropy and lower Gini coefficient indicate
        more balanced distributions.
    """
    dist = get_class_distribution(labels, normalize=True)
    
    metrics = {
        'imbalance_ratio': dist.max() / dist.min(),
        'entropy': -(dist * np.log(dist)).sum(),
        'gini': 1 - (dist ** 2).sum()
    }
    
    return metrics
    
def get_rare_label_stats(
    labels: Union[np.ndarray, pd.Series],
    threshold: float = 0.01,
    min_samples: int = 10
) -> Dict[str, float]:
    """Analyze rare classes in label data.
    
    This function identifies and analyzes rare classes:
    - Finds classes below frequency threshold
    - Computes statistics about rare classes
    - Provides detailed rare class metrics
    
    Args:
        labels: Input label array or series
        threshold: Frequency threshold for rare classes
        min_samples: Minimum samples to consider
        
    Returns:
        Dictionary of rare label statistics:
        - Number of rare classes
        - Proportion of rare classes
        - Sample counts and frequencies
        - Distribution metrics
        
    Note:
        Useful for identifying potential data collection
        or labeling issues.
    """
    dist = get_class_distribution(labels, normalize=True)
    rare_mask = (dist < threshold) & (dist * len(labels) <= min_samples)
    rare_dist = dist[rare_mask]
    
    stats = {
        'num_rare_labels': len(rare_dist),
        'rare_label_proportion': len(rare_dist) / len(dist),
        'rare_sample_proportion': rare_dist.sum(),
        'rare_label_list': list(rare_dist.index),
        'samples_per_rare': (rare_dist * len(labels)).mean(),
        'max_rare_samples': (rare_dist * len(labels)).max(),
        'min_rare_samples': (rare_dist * len(labels)).min()
    }
    
    if len(rare_dist) > 0:
        stats.update({
            'rarest_label': rare_dist.index[0],
            'rarest_frequency': rare_dist.min(),
            'rare_entropy': -(rare_dist * np.log(rare_dist)).sum(),
            'rare_gini': 1 - (rare_dist ** 2).sum()
        })
        
    return stats
    

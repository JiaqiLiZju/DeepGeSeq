"""Dataset module for genomic data processing.

This module provides:
1. Base dataset classes for genomic data
2. Data loading and preprocessing utilities
3. Common genomic data formats support
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
import logging

from .Interval import (
    Interval, NamedInterval
)
from .Sequence import (
    DNASeq, Genome,
    one_hot_encode, reverse_complement,
    calculate_gc_content, calculate_complexity
)
from .Target import Target

logger = logging.getLogger("dgs.dataset")

# Dataset Classes
class SeqDataset(Dataset):
    """Dataset for genomic sequences extracted based on intervals."""
    
    def __init__(self, intervals: Interval, genome: Genome, strand_aware: bool = True):
        """Initialize SeqDataset.
        
        Args:
            intervals: Interval object with intervals (chrom, start, end, strand)
            genome: Genome object for sequence extraction
            strand_aware: Whether to respect strand information
        """
        self.intervals = intervals
        self.genome = genome
        self.strand_aware = strand_aware
        self.seqs = self._extract_sequences()
        
    def _extract_sequences(self) -> List[DNASeq]:
        """Extract sequences from genome based on intervals."""
        return self.genome.extract_sequences(self.intervals.data, strand_aware=self.strand_aware)
        
    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.seqs)
        
    def __getitem__(self, idx: int) -> np.ndarray:
        """Return a sequence by index, encoded as one-hot."""
        seq = self.seqs[idx]
        return one_hot_encode(seq.sequence)

class GenomicDataset(SeqDataset):
    """Dataset for genomic intervals with labels.
    
    Features:
    - Use Intervals to manage sample position information
    - Efficient sequence and label storage
    - Flexible data loading strategy
    - Support multi-task learning
    - Automatic sequence encoding
    """
    
    def __init__(
        self,
        intervals: Interval,
        genome: Genome,
        targets: Target,
        strand_aware: bool = True
    ):
        """Initialize genomic dataset.
        
        Args:
            intervals: Interval object with intervals (chrom, start, end, strand)
            genome: Genome object for sequence extraction
            targets: Target object for label extraction
            strand_aware: Whether to respect strand information
        """
        super().__init__(intervals, genome, strand_aware)
        self.targets = targets
        self.labels = self.targets.get_labels()
        self.task_info = self.targets.get_task_info()
        
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return a sequence and its labels by index."""
        seq = self.seqs[idx]
        label = self.labels[idx]
        return one_hot_encode(seq.sequence), label.astype(np.float32)

# DataLoader Utilities
def create_dataloader(
    dataset: Union[SeqDataset, GenomicDataset],
    batch_size: Optional[int] = 4,
    shuffle: bool = True,
    **kwargs
) -> torch.utils.data.DataLoader:
    """Create DataLoader for dataset.
    
    Args:
        dataset: Input dataset
        batch_size: Batch size (uses dataset config if None)
        shuffle: Whether to shuffle data
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        DataLoader instance
    """
        
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )

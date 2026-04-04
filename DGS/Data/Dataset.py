"""Dataset wrappers used by DGS training and inference pipelines.

Purpose:
    Provide dataset classes that expose one-hot DNA inputs and optional labels.

Main Responsibilities:
    - Build sequence-only datasets (`SeqDataset`) from intervals and genome FASTA.
    - Build supervised datasets (`GenomicDataset`) that pair sequences and targets.
    - Create PyTorch dataloaders with backward-compatible worker/runtime options.

Key Runtime Notes:
    - Single-sample sequence shape is `(sequence_length, 4)` before collation.
    - Labels are cast to `float32` in supervised dataset outputs.
    - Dataloader worker options are only enabled when `num_workers > 0`.
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

class SeqDataset(Dataset):
    """Dataset class for genomic sequence data.
    
    This class provides:
    - Efficient sequence extraction from genome
    - Automatic one-hot encoding
    - Strand-aware processing
    - Memory-efficient operations
    
    Attributes:
        intervals: Genomic intervals for sequence extraction.
        genome: Reference genome object.
        strand_aware: Whether to respect strand information.
        seqs: Cached sequence objects.
        
    Example:
        >>> genome = Genome("hg38.fa")
        >>> intervals = Interval("regions.bed")
        >>> dataset = SeqDataset(intervals, genome)
        >>> seq = dataset[0]  # Get first sequence
    """
    
    def __init__(self, intervals: Interval, genome: Genome, strand_aware: bool = True):
        """Initialize sequence dataset.
        
        Args:
            intervals: Interval object with genomic regions
            genome: Genome object for sequence extraction
            strand_aware: Whether to respect strand information
        """
        self.intervals = intervals
        self.genome = genome
        self.strand_aware = strand_aware
        self.seqs = self._extract_sequences()
        
    def _extract_sequences(self) -> List[DNASeq]:
        """Extract DNA sequences from genome.
        
        Returns:
            List of `DNASeq` objects corresponding to intervals.
            
        Note:
            This method caches sequences to avoid repeated
            genome access operations.
        """
        return self.genome.extract_sequences(self.intervals.data, strand_aware=self.strand_aware)
        
    def __len__(self) -> int:
        """Return the number of sequences in dataset."""
        return len(self.seqs)
        
    def __getitem__(self, idx: int) -> np.ndarray:
        """Get one-hot encoded sequence by index.
        
        Args:
            idx: Index of sequence to retrieve.
            
        Returns:
            One-hot encoded sequence as a numpy array with shape
            `(sequence_length, 4)`.
        """
        seq = self.seqs[idx]
        return one_hot_encode(seq.sequence)

class GenomicDataset(SeqDataset):
    """Dataset class for genomic sequences with associated labels.
    
    This class extends SeqDataset with:
    - Multi-task label support
    - Flexible data loading
    - Efficient storage
    - Comprehensive metadata
    
    Features:
    - Use Intervals for position management
    - Efficient sequence/label storage
    - Support for multi-task learning
    - Automatic sequence encoding
    
    Attributes:
        intervals: Genomic intervals
        genome: Reference genome
        targets: Target data manager
        labels: Cached label arrays
        task_info: Task configuration data
        
    Example:
        >>> genome = Genome("hg38.fa")
        >>> intervals = Interval("regions.bed")
        >>> targets = Target("labels.bed")
        >>> dataset = GenomicDataset(intervals, genome, targets)
        >>> seq, label = dataset[0]  # Get first sequence and labels
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
            intervals: Interval object with regions
            genome: Genome object for sequences
            targets: Target object with labels
            strand_aware: Whether to respect strand
        """
        super().__init__(intervals, genome, strand_aware)
        self.targets = targets
        self.labels = self.targets.get_labels()
        self.task_info = self.targets.get_task_info()
        
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get sequence and labels by index.
        
        Args:
            idx: Index of sample to retrieve.
            
        Returns:
            Tuple `(sequence, labels)` where:
                - `sequence` has shape `(sequence_length, 4)`.
                - `labels` contains task values for the sample.
        """
        seq = self.seqs[idx]
        label = self.labels[idx]
        return one_hot_encode(seq.sequence), label.astype(np.float32)

def create_dataloader(
    dataset: Union[SeqDataset, GenomicDataset],
    batch_size: Optional[int] = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
    **kwargs
) -> torch.utils.data.DataLoader:
    """Create PyTorch DataLoader for genomic datasets.
    
    This function provides:
    - Efficient batch generation
    - Optional shuffling
    - Customizable loading
    - Memory management
    
    Args:
        dataset: Input genomic dataset
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle samples
        num_workers: Number of subprocesses for loading data
        pin_memory: Whether to pin memory for faster host->device transfer
        persistent_workers: Keep workers alive between epochs
        prefetch_factor: Number of batches preloaded per worker
        **kwargs: Additional DataLoader arguments
        
    Returns:
        PyTorch `DataLoader` configured for the dataset.
        
    Example:
        >>> dataset = GenomicDataset(intervals, genome, targets)
        >>> loader = create_dataloader(dataset, batch_size=32)
        >>> for sequences, labels in loader:
        ...     # Training loop
    """
    dataloader_kwargs = dict(kwargs)
    dataloader_kwargs["num_workers"] = int(num_workers)
    dataloader_kwargs["pin_memory"] = bool(pin_memory)
    if dataloader_kwargs["num_workers"] > 0:
        dataloader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **dataloader_kwargs
    )

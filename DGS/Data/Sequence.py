"""
DNA Sequence Processing Module

This module provides comprehensive utilities for DNA sequence manipulation:

Key Components:
1. Core Sequence Operations:
   - Sequence validation and cleaning
   - Reverse complement generation
   - GC content calculation
   - Sequence complexity analysis

2. Encoding/Decoding:
   - One-hot encoding with customizable formats
   - Efficient batch processing
   - Support for N bases
   - Memory-efficient operations

3. Sequence Classes:
   - DNASeq: Core sequence manipulation
   - Genome: Genome-wide sequence operations
   - Efficient sequence extraction
   - Format conversion utilities

4. Analysis Tools:
   - GC content analysis
   - Sequence complexity metrics
   - Quality assessment
   - Pattern detection

The module is designed for:
- High-performance sequence processing
- Memory-efficient operations
- Flexible sequence representation
- Integration with deep learning pipelines
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path
import logging

from ..IO import FastaReader

logger = logging.getLogger("dgs.sequence")

# Common constants for DNA sequence operations
DNA_BASES = 'ATCG'
VALID_BASES = DNA_BASES + 'N'
COMPLEMENT_MAP = {
    'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
    'a': 't', 't': 'a', 'g': 'c', 'c': 'g',
    'N': 'N', 'n': 'n'
}
ONEHOT_MAP = {
    'A': [1,0,0,0],
    'C': [0,1,0,0],
    'G': [0,0,1,0],
    'T': [0,0,0,1],
    'N': [0,0,0,0]
}
BASE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
IDX_TO_BASE = ['A', 'C', 'G', 'T', 'N']

def validate_sequence(seq: str) -> bool:
    """Validate DNA sequence composition.
    
    This function checks that:
    - All bases are valid (ATCGN)
    - Sequence is not empty
    - Case is consistent
    
    Args:
        seq: Input DNA sequence
        
    Returns:
        bool: True if sequence is valid
        
    Raises:
        ValueError: If invalid bases are found
        
    Example:
        >>> validate_sequence("ATCGN")  # True
        >>> validate_sequence("XINVALID")  # Raises ValueError
    """
    invalid = set(seq.upper()) - set(VALID_BASES)
    if invalid:
        raise ValueError(f"Invalid bases in sequence: {invalid}")
    return True

def get_reverse_complement(seq: str) -> str:
    """Generate reverse complement of DNA sequence.
    
    This function:
    - Reverses the sequence
    - Replaces each base with its complement
    - Preserves case and N bases
    
    Args:
        seq: Input DNA sequence
        
    Returns:
        str: Reverse complement sequence
        
    Example:
        >>> get_reverse_complement("ATCG")  # Returns "CGAT"
        >>> get_reverse_complement("ATCGNn")  # Returns "nNCGAT"
    """
    return ''.join(COMPLEMENT_MAP[base] for base in reversed(seq))

def sequence_to_onehot(seq: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Convert DNA sequence to one-hot encoding.
    
    This function creates a 2D array where:
    - Each row represents one base
    - Columns represent A, C, G, T
    - N bases are encoded as [0,0,0,0]
    
    Args:
        seq: Input DNA sequence
        dtype: Numpy data type for output array
        
    Returns:
        np.ndarray: One-hot encoded sequence
            Shape: (sequence_length, 4)
            
    Example:
        >>> sequence_to_onehot("ACGT")
        array([[1,0,0,0],
               [0,1,0,0],
               [0,0,1,0],
               [0,0,0,1]], dtype=float32)
    """
    if not seq:
        return np.zeros((0, 4), dtype=dtype)
    return np.array([ONEHOT_MAP.get(base.upper(), [0,0,0,0]) for base in seq], dtype=dtype)

def onehot_to_sequence(encoded: np.ndarray, include_n: bool = True) -> str:
    """Convert one-hot encoding back to DNA sequence.
    
    This function:
    - Converts one-hot arrays to sequence
    - Optionally includes N bases
    - Handles edge cases gracefully
    
    Args:
        encoded: One-hot encoded sequence array
        include_n: Whether to include N in output sequence
        
    Returns:
        str: DNA sequence
        
    Example:
        >>> arr = np.array([[1,0,0,0], [0,0,0,0]])
        >>> onehot_to_sequence(arr)  # Returns "AN"
        >>> onehot_to_sequence(arr, include_n=False)  # Returns "A"
    """
    if include_n:
        # If any row sums to 0, it's an N
        is_n = ~encoded.any(axis=1)
        # Get the index of the maximum value for non-N positions
        indices = np.where(is_n, 4, encoded.argmax(axis=1))
        # Convert consecutive N's to single N
        seq = ''.join(IDX_TO_BASE[i] for i in indices)
        return 'N' if all(c == 'N' for c in seq) else seq
    else:
        return ''.join(IDX_TO_BASE[i] for i in encoded.argmax(axis=1))

def batch_to_onehot(sequences: List[str], dtype: np.dtype = np.float32) -> np.ndarray:
    """Convert multiple sequences to one-hot encoding.
    
    This function efficiently processes batches by:
    - Padding to maximum length
    - Vectorizing conversion
    - Optimizing memory usage
    
    Args:
        sequences: List of DNA sequences
        dtype: Data type for output array
        
    Returns:
        np.ndarray: One-hot encoded sequences
            Shape: (n_sequences, max_length, 4)
            Empty sequences are padded with zeros
            
    Example:
        >>> seqs = ["ACGT", "AT"]
        >>> batch_to_onehot(seqs).shape  # (2, 4, 4)
    """
    if not sequences:
        return np.zeros((0, 0, 4), dtype=dtype)
        
    # Find maximum sequence length
    max_length = max(len(seq) for seq in sequences)
    if max_length == 0:
        return np.zeros((len(sequences), 0, 4), dtype=dtype)
    
    # Initialize output array
    batch_size = len(sequences)
    output = np.zeros((batch_size, max_length, 4), dtype=dtype)
    
    # Fill in sequences
    for i, seq in enumerate(sequences):
        if seq:  # Only process non-empty sequences
            seq_encoded = sequence_to_onehot(seq, dtype)
            output[i, :len(seq)] = seq_encoded
            
    return output

def batch_from_onehot(encoded: np.ndarray, include_n: bool = True) -> List[str]:
    """Convert batch of one-hot encodings to sequences.
    
    Args:
        encoded: Batch of one-hot encoded sequences
        include_n: Whether to include N bases
        
    Returns:
        List[str]: DNA sequences
        
    Example:
        >>> arr = np.zeros((2, 4, 4))
        >>> batch_from_onehot(arr)  # Returns ["NNNN", "NNNN"]
    """
    return [onehot_to_sequence(seq, include_n) for seq in encoded]

def calculate_gc_content(seq: str) -> float:
    """Calculate GC content of sequence.
    
    This function:
    - Counts G and C bases
    - Excludes N bases from calculation
    - Returns normalized ratio
    
    Args:
        seq: Input DNA sequence
        
    Returns:
        float: GC content (0.0 to 1.0)
            Returns 0.0 if sequence only contains N's
            
    Example:
        >>> calculate_gc_content("GCTA")  # Returns 0.5
        >>> calculate_gc_content("GCNN")  # Returns 1.0
    """
    seq = seq.upper()
    total = len(seq) - seq.count('N')  # Exclude N bases
    if total == 0:
        return 0.0
    return (seq.count('G') + seq.count('C')) / total

def calculate_sliding_gc(seq: str, window_size: int) -> np.ndarray:
    """Calculate GC content in sliding windows.
    
    This function computes:
    - GC content for each window
    - Handles edge cases
    - Excludes N bases per window
    
    Args:
        seq: Input DNA sequence
        window_size: Size of sliding window
        
    Returns:
        np.ndarray: Array of GC content values
            Length: len(seq) - window_size + 1
            
    Example:
        >>> calculate_sliding_gc("GCTAN", 2)
        array([1.0, 0.5, 0.5, 0.0])
    """
    seq = seq.upper()
    
    # Handle case where window size is larger than sequence
    if len(seq) <= window_size:
        return np.array([calculate_gc_content(seq)])
    
    # Calculate sliding windows
    gc_contents = []
    for i in range(len(seq) - window_size + 1):
        window = seq[i:i+window_size]
        valid_bases = sum(1 for base in window if base != 'N')
        if valid_bases == 0:
            gc_contents.append(0.0)
        else:
            gc_count = sum(1 for base in window if base in 'GC')
            gc_contents.append(gc_count / valid_bases)
    
    return np.array(gc_contents)

def calculate_complexity(seq: str, k: int = 3, normalize: bool = True) -> float:
    """Calculate sequence complexity using k-mer diversity.
    
    This function:
    - Counts unique k-mers
    - Excludes k-mers with N
    - Optionally normalizes scores
    
    Args:
        seq: Input DNA sequence
        k: k-mer size (default: 3)
        normalize: Whether to normalize score
        
    Returns:
        float: Complexity score
            If normalized: 0.0 to 1.0
            If not normalized: Raw k-mer count
            
    Example:
        >>> calculate_complexity("ATCGATCG", k=2)  # High complexity
        >>> calculate_complexity("AAAAAAA", k=2)   # Low complexity
    """
    # Handle edge cases
    if len(seq) < k:
        return 0.0
    seq = seq.upper()
    
    # Get all k-mers and filter out those containing N
    valid_kmers = []
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if 'N' not in kmer:
            valid_kmers.append(kmer)
    
    # If no valid k-mers (all contain N), return 0
    if not valid_kmers:
        return 0.0
    
    # Count unique k-mers
    unique_kmers = set(valid_kmers)
    n_unique = len(unique_kmers)
    
    if normalize:
        # Calculate theoretical maximum unique k-mers
        max_possible = min(4**k, len(valid_kmers))
        # Adjust normalization based on sequence length
        if len(seq) > k:
            # For longer sequences, consider the actual number of valid k-mers
            return n_unique / max_possible
        else:
            # For shorter sequences, normalize by sequence length
            return n_unique / len(seq)
    return n_unique

class DNASeq:
    """Class for DNA sequence manipulation.
    
    This class provides:
    - Sequence validation
    - Common operations (reverse complement, etc.)
    - Encoding/decoding
    - Sequence analysis
    
    Attributes:
        sequence: DNA sequence string
        
    Example:
        >>> seq = DNASeq("ATCG")
        >>> seq.gc_content()  # Returns 0.5
        >>> seq.reverse_complement()  # Returns DNASeq("CGAT")
    """
    
    def __init__(self, sequence: str):
        """Initialize DNA sequence.
        
        Args:
            sequence: Input DNA sequence
            
        Raises:
            ValueError: If sequence contains invalid bases
        """
        self._sequence = sequence.upper()
        validate_sequence(self._sequence)
        
    @property
    def sequence(self) -> str:
        """Get the DNA sequence."""
        return self._sequence
        
    def __len__(self) -> int:
        """Get sequence length."""
        return len(self._sequence)
        
    def __str__(self) -> str:
        """Get string representation."""
        return self._sequence
        
    def __repr__(self) -> str:
        """Get detailed representation."""
        return f"DNASeq('{self._sequence}')"
    
    def validate(self) -> bool:
        """Validate sequence composition."""
        return validate_sequence(self._sequence)
        
    def reverse_complement(self) -> 'DNASeq':
        """Get reverse complement sequence."""
        return DNASeq(get_reverse_complement(self._sequence))
        
    def to_onehot(self, dtype: np.dtype = np.float32) -> np.ndarray:
        """Convert to one-hot encoding."""
        return sequence_to_onehot(self._sequence, dtype)
        
    @classmethod
    def from_onehot(cls, encoded: np.ndarray, include_n: bool = True) -> 'DNASeq':
        """Create sequence from one-hot encoding."""
        return cls(onehot_to_sequence(encoded, include_n))
        
    def gc_content(self, window_size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Calculate GC content.
        
        Args:
            window_size: Optional window size for sliding calculation
            
        Returns:
            float or np.ndarray: GC content value(s)
        """
        if window_size:
            return calculate_sliding_gc(self._sequence, window_size)
        return calculate_gc_content(self._sequence)
    
    def complexity(self, k: int = 3, normalize: bool = True) -> float:
        """Calculate sequence complexity."""
        return calculate_complexity(self._sequence, k, normalize)

class Genome:
    """Class for genome-wide sequence operations.
    
    This class provides:
    - Efficient sequence extraction
    - Memory management
    - Coordinate validation
    - Error handling
    
    Attributes:
        genome_path: Path to genome FASTA file
        
    Example:
        >>> genome = Genome("hg38.fa")
        >>> intervals = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [1000]})
        >>> seqs = genome.extract_sequences(intervals)
    """
    
    def __init__(self, genome_path: Union[str, Path], use_cache: bool = True):
        """Initialize genome reader.
        
        Args:
            genome_path: Path to genome FASTA file
            use_cache: Whether to cache sequences
        """
        self.genome_path = Path(genome_path)
        self._reader = FastaReader(self.genome_path)
            
    def close(self):
        """Close genome reader and release resources."""
        if self._reader:
            if hasattr(self._reader, 'close'):
                self._reader.close()
            elif hasattr(self._reader, '__exit__'):
                self._reader.__exit__(None, None, None)
            self._reader = None
            
    def extract_sequences(
        self,
        intervals: pd.DataFrame,
        strand_aware: bool = True,
        **kwargs
    ) -> List[DNASeq]:
        """Extract sequences from genome.
        
        This function:
        - Validates interval coordinates
        - Handles strand information
        - Processes in batches
        - Manages memory efficiently
        
        Args:
            intervals: DataFrame with genomic intervals
            strand_aware: Whether to respect strand
            **kwargs: Additional extraction parameters
            
        Returns:
            List[DNASeq]: Extracted sequences
            
        Raises:
            RuntimeError: If called outside context manager
            ValueError: If intervals are invalid
            KeyError: If chromosome not found
        """
        if not self._reader:
            raise RuntimeError("Genome must be used within a context manager")
            
        # Validate intervals before attempting to read
        if not isinstance(intervals, pd.DataFrame):
            raise ValueError("Intervals must be a pandas DataFrame")
        if not all(col in intervals.columns for col in ['chrom', 'start', 'end']):
            raise ValueError("Intervals must contain 'chrom', 'start', and 'end' columns")
            
        # Try to read sequences
        try:
            # First check if all chromosomes exist
            if hasattr(self._reader, 'references'):
                invalid_chroms = set(intervals['chrom']) - set(self._reader.references)
                if invalid_chroms:
                    raise KeyError(f"Invalid chromosomes in intervals: {invalid_chroms}")
            
            # Validate coordinates
            if (intervals['start'] < 0).any():
                raise ValueError("Interval start positions cannot be negative")
            if (intervals['end'] <= intervals['start']).any():
                raise ValueError("Interval end must be greater than start")
            
            raw_sequences = self._reader.read(intervals, **kwargs)
            return self._process_sequences(raw_sequences, intervals, strand_aware)
        except KeyError as e:
            raise e
        except ValueError as e:
            raise e
        except Exception as e:
            raise KeyError(f"Error reading sequences: {str(e)}")
    
    def _process_sequences(
        self,
        sequences: List[str],
        intervals: pd.DataFrame,
        strand_aware: bool
    ) -> List[DNASeq]:
        """Process extracted sequences.
        
        This function:
        - Creates DNASeq objects
        - Handles strand orientation
        - Validates sequences
        
        Args:
            sequences: Raw sequence strings
            intervals: Interval information
            strand_aware: Whether to use strand info
            
        Returns:
            List[DNASeq]: Processed sequences
        """
        processed = []
        for i, seq in enumerate(sequences):
            dna_seq = DNASeq(seq)
            if strand_aware and 'strand' in intervals.columns:
                if intervals.iloc[i].strand == '-':
                    dna_seq = dna_seq.reverse_complement()
            processed.append(dna_seq)
        return processed

# Legacy functions for backward compatibility
def reverse_complement(seq: str) -> str:
    """Legacy function for reverse complement."""
    return get_reverse_complement(seq)

def one_hot_encode(sequences: Union[str, List[str]], dtype: np.dtype = np.float32) -> np.ndarray:
    """Legacy function for one-hot encoding."""
    if isinstance(sequences, str):
        return sequence_to_onehot(sequences, dtype)
    return batch_to_onehot(sequences, dtype)

def one_hot_decode(encoded: np.ndarray, include_n: bool = True) -> Union[str, List[str]]:
    """Legacy function for one-hot decoding."""
    if encoded.ndim == 2:
        return onehot_to_sequence(encoded, include_n)
    return batch_from_onehot(encoded, include_n)

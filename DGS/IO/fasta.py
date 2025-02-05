"""FASTA file reader module for DGS.

This module provides:
1. FastaReader - Reader for FASTA sequence files
2. Efficient sequence extraction by coordinates
3. Support for genome indexing
4. Batch sequence processing
"""

import pandas as pd
from typing import List, Optional, Union, Dict, Iterator
from pathlib import Path
import pysam

from . import logger

class FastaReader:
    """Reader for FASTA sequence files.
    
    Features:
    - Random access to genomic sequences
    - Efficient sequence extraction
    - Coordinate validation
    - Batch processing
    """
    
    VALID_EXTENSIONS = [".fa", ".fasta", ".fa.gz", ".fasta.gz"]
    
    def __init__(
        self,
        file_path: Union[str, Path],
        validate_index: bool = True
    ):
        """Initialize FASTA reader.
        
        Args:
            file_path: Path to FASTA file
            validate_index: Whether to validate index file
        """
        self.file_path = Path(file_path)
        self._fasta = None
        
        # Check if file exists before validation
        if not self.check_file():
            raise ValueError(f"Invalid file: {self.file_path}")
            
        # Validate index
        if validate_index:
            self._check_index()
            
    def check_file(self) -> bool:
        """Check if file has valid extension and exists."""
        # Get the base extension without .gz
        file_path_str = str(self.file_path)
        if file_path_str.endswith('.gz'):
            base_ext = Path(file_path_str[:-3]).suffix
        else:
            base_ext = self.file_path.suffix
            
        return (
            base_ext in self.VALID_EXTENSIONS
            and self.file_path.exists()
        )
            
    def _check_index(self):
        """Check if index file exists and create if needed."""
        fai_path = str(self.file_path) + '.fai'
        if not Path(fai_path).exists():
            pysam.faidx(str(self.file_path))
            
    def __enter__(self):
        if self._fasta is None:
            self._fasta = pysam.FastaFile(str(self.file_path))
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fasta is not None:
            self._fasta.close()
            self._fasta = None
            
    @property
    def references(self) -> List[str]:
        """Get list of reference sequences."""
        if self._fasta is None:
            self._fasta = pysam.FastaFile(str(self.file_path))
        return self._fasta.references
        
    @property
    def lengths(self) -> Dict[str, int]:
        """Get dictionary of reference lengths."""
        if self._fasta is None:
            self._fasta = pysam.FastaFile(str(self.file_path))
        return dict(zip(self._fasta.references, self._fasta.lengths))
            
    def read(
        self,
        intervals: Optional[pd.DataFrame] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Union[Dict[str, str], List[str], Iterator[List[str]]]:
        """Read sequences from FASTA file.
        
        Args:
            intervals: Optional DataFrame with genomic intervals
            batch_size: Optional batch size for iterative processing
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of sequences, list of sequences, or sequence iterator
        """
        if self._fasta is None:
            self._fasta = pysam.FastaFile(str(self.file_path))
                
        if intervals is None:
            # Read whole sequences
            return {
                ref: self._fasta.fetch(ref)
                for ref in self._fasta.references
            }
                
        # Validate intervals
        self.validate_intervals(intervals)
            
        # Handle batch processing
        if batch_size:
            return self._batch_extract(intervals, batch_size)
                
        # Extract sequences
        return self._extract_sequences(intervals)
            
    def _batch_extract(
        self,
        intervals: pd.DataFrame,
        batch_size: int
    ) -> Iterator[List[str]]:
        """Extract sequences in batches."""
        for i in range(0, len(intervals), batch_size):
            batch = intervals.iloc[i:i+batch_size]
            yield self._extract_sequences(batch)
            
    def _extract_sequences(self, intervals: pd.DataFrame) -> List[str]:
        """Extract sequences for a set of intervals."""
        sequences = []
        for _, row in intervals.iterrows():
            try:
                seq = self.get_sequence(
                    row.chrom,
                    row.start,
                    row.end
                )
                sequences.append(seq)
            except Exception as e:
                sequences.append("N" * (row.end - row.start))
        return sequences
            
    def get_sequence(
        self,
        chrom: str,
        start: int,
        end: int
    ) -> str:
        """Get sequence for specific coordinates.
        
        Args:
            chrom: Chromosome name
            start: Start position (0-based)
            end: End position
            
        Returns:
            DNA sequence
            
        Raises:
            ValueError: If coordinates are invalid
        """
        # Validate coordinates
        if start < 0:
            raise ValueError(f"Negative start position: {start}")
        if end <= start:
            raise ValueError(f"Invalid interval: end ({end}) <= start ({start})")
        if chrom not in self.references:
            raise ValueError(f"Unknown chromosome: {chrom}")
                
        # Get sequence
        return self._fasta.fetch(chrom, start, end)
        
    def validate_intervals(self, intervals: pd.DataFrame) -> None:
        """Validate genomic intervals.
        
        Args:
            intervals: DataFrame with genomic intervals
            
        Raises:
            ValueError: If intervals are invalid
        """
        # Check required columns
        required_cols = ['chrom', 'start', 'end']
        missing_cols = [col for col in required_cols if col not in intervals.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Validate data types
        if not intervals['chrom'].dtype == object:  # Chromosome names should be strings
            raise ValueError("'chrom' column must be string type")
        if not pd.api.types.is_numeric_dtype(intervals['start']):
            raise ValueError("'start' column must be numeric type")
        if not pd.api.types.is_numeric_dtype(intervals['end']):
            raise ValueError("'end' column must be numeric type")
            
        # Validate interval coordinates
        if (intervals['end'] <= intervals['start']).any():
            raise ValueError("Interval end positions must be greater than start positions")
        if (intervals['start'] < 0).any():
            raise ValueError("Interval start positions cannot be negative")

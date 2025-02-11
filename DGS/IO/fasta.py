"""FASTA file reader module for genomic sequence data.

This module provides functionality for reading and processing FASTA format files,
which store DNA/RNA sequence data. It supports random access to sequences and
efficient extraction of genomic regions.

Classes
-------
FastaReader
    Reader for FASTA sequence files with the following features:
    - Random access to genomic sequences
    - Efficient sequence extraction by coordinates
    - Support for indexed FASTA files
    - Batch sequence processing
    - Memory-efficient reading
    - Automatic index creation

Notes
-----
The module uses pysam for efficient FASTA file handling and requires:
- FASTA files (.fa, .fasta)
- Optional compression (.gz)
- Index files (.fai) for random access
- Proper sequence formatting (no line wrapping recommended)

Examples
--------
>>> reader = FastaReader("genome.fa")
>>> # Get specific region
>>> seq = reader.get_sequence("chr1", 1000, 2000)
>>> # Get multiple regions
>>> intervals = pd.DataFrame({
...     "chrom": ["chr1", "chr2"],
...     "start": [1000, 2000],
...     "end": [1500, 2500]
... })
>>> seqs = reader.read(intervals)
"""

import pandas as pd
from typing import List, Optional, Union, Dict, Iterator
from pathlib import Path
import pysam

from . import logger

class FastaReader:
    """Reader for FASTA sequence files.
    
    Provides functionality for reading and extracting sequences from FASTA format
    files, with support for random access and batch processing.

    Attributes
    ----------
    file_path : Path
        Path to the FASTA file
    VALID_EXTENSIONS : list of str
        List of valid file extensions [".fa", ".fasta", ".fa.gz", ".fasta.gz"]
    _fasta : pysam.FastaFile or None
        Pysam FASTA file handle

    Methods
    -------
    read(intervals=None, batch_size=None, **kwargs)
        Read sequences from specified intervals
    get_sequence(chrom, start, end)
        Get sequence for specific coordinates
    check_file()
        Validate file path and extension
    validate_intervals(intervals)
        Validate genomic interval data format

    Notes
    -----
    - Uses context manager for proper resource handling
    - Automatically creates index file if missing
    - Supports both whole genome and interval-based reading
    - Memory-efficient batch processing for large datasets
    """
    
    VALID_EXTENSIONS = [".fa", ".fasta", ".fa.gz", ".fasta.gz"]
    
    def __init__(
        self,
        file_path: Union[str, Path],
        validate_index: bool = True
    ):
        """Initialize FASTA reader.
        
        Parameters
        ----------
        file_path : str or Path
            Path to FASTA file
        validate_index : bool, optional
            Whether to validate and create index file if missing (default: True)
            
        Raises
        ------
        ValueError
            If file path is invalid or file doesn't exist
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
        """Check if file has valid extension and exists.
        
        Returns
        -------
        bool
            True if file is valid, False otherwise
            
        Notes
        -----
        Handles both gzipped and uncompressed files by checking the
        base extension before .gz if present.
        """
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
        """Check if index file exists and create if needed.
        
        Notes
        -----
        Creates .fai index file using pysam.faidx if not present.
        Index file is required for random access to sequences.
        """
        fai_path = str(self.file_path) + '.fai'
        if not Path(fai_path).exists():
            pysam.faidx(str(self.file_path))
            
    def __enter__(self):
        """Context manager entry.
        
        Returns
        -------
        FastaReader
            Self reference for context manager
        """
        if self._fasta is None:
            self._fasta = pysam.FastaFile(str(self.file_path))
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit.
        
        Ensures proper cleanup of FASTA file handle.
        """
        if self._fasta is not None:
            self._fasta.close()
            self._fasta = None
            
    @property
    def references(self) -> List[str]:
        """Get list of reference sequences.
        
        Returns
        -------
        list of str
            Names of all sequences in the FASTA file
        """
        if self._fasta is None:
            self._fasta = pysam.FastaFile(str(self.file_path))
        return self._fasta.references
        
    @property
    def lengths(self) -> Dict[str, int]:
        """Get dictionary of reference lengths.
        
        Returns
        -------
        dict
            Mapping of sequence names to their lengths
        """
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
        
        Parameters
        ----------
        intervals : pd.DataFrame, optional
            DataFrame with genomic intervals (chrom, start, end)
        batch_size : int, optional
            Size of batches for iterative processing
        **kwargs : dict
            Additional arguments for sequence extraction
            
        Returns
        -------
        Union[Dict[str, str], List[str], Iterator[List[str]]]
            - Dict mapping sequence names to sequences if no intervals
            - List of sequences if intervals provided
            - Iterator of sequence batches if batch_size specified
            
        Notes
        -----
        - Without intervals, returns all sequences
        - With intervals, extracts specific regions
        - With batch_size, processes intervals in chunks
        - Missing regions return N's of appropriate length
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
        """Extract sequences in batches.
        
        Parameters
        ----------
        intervals : pd.DataFrame
            DataFrame with genomic intervals
        batch_size : int
            Number of intervals per batch
            
        Returns
        -------
        Iterator[List[str]]
            Iterator yielding lists of sequences
            
        Notes
        -----
        Memory-efficient processing for large interval sets.
        """
        for i in range(0, len(intervals), batch_size):
            batch = intervals.iloc[i:i+batch_size]
            yield self._extract_sequences(batch)
            
    def _extract_sequences(self, intervals: pd.DataFrame) -> List[str]:
        """Extract sequences for a set of intervals.
        
        Parameters
        ----------
        intervals : pd.DataFrame
            DataFrame with genomic intervals
            
        Returns
        -------
        list of str
            List of extracted sequences
            
        Notes
        -----
        Returns N's for invalid or missing regions.
        """
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
        
        Parameters
        ----------
        chrom : str
            Chromosome or sequence name
        start : int
            Start position (0-based)
        end : int
            End position
            
        Returns
        -------
        str
            DNA sequence
            
        Raises
        ------
        ValueError
            If coordinates are invalid
            
        Notes
        -----
        - Coordinates are 0-based, half-open [start, end)
        - Validates coordinates before extraction
        - Requires chromosome to exist in references
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
        
        Parameters
        ----------
        intervals : pd.DataFrame
            DataFrame with genomic intervals
            
        Raises
        ------
        ValueError
            If intervals are invalid
            
        Notes
        -----
        Validates:
        - Required columns (chrom, start, end)
        - Column data types (chrom: str, start/end: numeric)
        - Coordinate ordering (end > start)
        - Non-negative coordinates
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

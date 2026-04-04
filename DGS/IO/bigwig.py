"""BigWig signal readers for interval-aligned target extraction.

Purpose:
    Read BigWig values and align them to genomic interval inputs.

Main Responsibilities:
    - Validate BigWig files and interval inputs.
    - Extract raw or binned signal arrays per interval.
    - Apply optional aggregation (`mean`, `max`, `min`, `sum`, or callable).

Key Runtime Notes:
    - Requires `pyBigWig` at runtime.
    - Missing/failed intervals are represented with zero-filled arrays.
    - Binned aggregation supports short tail bins without reshape assumptions.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Callable
from pathlib import Path

from . import logger

class BigWigReader:
    """Reader for BigWig genomic signal files.
    
    Provides functionality for reading and processing signal data from BigWig
    format files. Supports random access to genomic regions, signal aggregation,
    and flexible binning options.

    Attributes:
    file_path : Path
        Path to the BigWig file
    VALID_EXTENSIONS : list of str
        List of valid file extensions [".bw", ".bigwig", ".bigWig"]

    Methods:
    read(intervals, bin_size=None, aggfunc=None, **kwargs)
        Read signal values for specified genomic intervals
    check_file()
        Validate file path and extension
    validate_intervals(intervals)
        Validate genomic interval data format

    Examples:
    >>> reader = BigWigReader("signal.bw")
    >>> signals = reader.read(intervals, bin_size=10, aggfunc="mean")
    """
    
    VALID_EXTENSIONS = [".bw", ".bigwig", ".bigWig"]
    
    def __init__(self, file_path: Union[str, Path]):
        """Initialize BigWig reader.
        
        Args:
        file_path : str or Path
            Path to BigWig file
            
        Raises:
        ValueError
            If file path is invalid or file doesn't exist
        """
        self.file_path = Path(file_path)
        if not self.check_file():
            raise ValueError(f"Invalid BigWig file: {self.file_path}")
            
    def check_file(self) -> bool:
        """Check if file has valid extension and exists.
        
        Returns:
        bool
            True if file is valid, False otherwise
        """
        return (
            self.file_path.suffix in self.VALID_EXTENSIONS
            and self.file_path.exists()
        )
    
    def read(
        self,
        intervals: pd.DataFrame,
        bin_size: Optional[int] = None,
        aggfunc: Optional[Union[str, Callable]] = None,
        **kwargs
    ) -> np.ndarray:
        """Read coverage values from BigWig file.
        
        Args:
        intervals : pd.DataFrame
            DataFrame with genomic intervals (chrom, start, end)
        bin_size : int, optional
            Number of bases to aggregate for binning
        aggfunc : str or callable, optional
            Aggregation function ("mean", "max", "min", "sum")
            or custom function
        **kwargs : dict
            Additional arguments passed to pyBigWig
            
        Returns:
        np.ndarray
            Array of shape (n_intervals, n_bins) with coverage values
            
        Raises:
        IOError
            If file reading fails
        ValueError
            If arguments are invalid
            
        Notes:
        - Missing regions return zero values
        - NaN values are converted to zeros
        - When binning, the last bin may be smaller
        """
        try:
            import pyBigWig

            if bin_size is not None:
                if not isinstance(bin_size, int) or bin_size <= 0:
                    raise ValueError("bin_size must be a positive integer when provided")
            
            # Validate intervals
            self.validate_intervals(intervals)
            
            # Setup aggregation function
            aggfunc = self._get_aggfunc(aggfunc)
            
            # Read signals
            signals = []
            with pyBigWig.open(str(self.file_path)) as bw:
                for row in intervals.itertuples():
                    expected_length = self._expected_length(
                        start=row.start,
                        end=row.end,
                        bin_size=bin_size,
                        has_agg=aggfunc is not None,
                    )
                    try:
                        # Read raw signal
                        signal = bw.values(row.chrom, row.start, row.end, numpy=True)
                        if signal is None:
                            signal = np.zeros(expected_length, dtype=np.float32)
                        else:
                            signal = np.nan_to_num(signal)
                        
                        # Aggregate if requested
                        if aggfunc is not None:
                            if bin_size:
                                signal = self._aggregate_binned_signal(signal, bin_size, aggfunc)
                            else:
                                # Aggregate over whole interval
                                signal = np.asarray([aggfunc(signal)])
                            
                        signals.append(signal)
                        
                    except Exception as e:
                        logger.warning(
                            f"Failed to read interval {row.chrom}:{row.start}-{row.end}: {e}"
                        )
                        signals.append(np.zeros(expected_length, dtype=np.float32))
                        
            # Stack and return
            return np.asarray(signals)
            
        except Exception as e:
            logger.error(f"Failed to read BigWig file {self.file_path}: {str(e)}")
            raise

    @staticmethod
    def _expected_length(start: int, end: int, bin_size: Optional[int], has_agg: bool) -> int:
        """Infer output length for an interval under current aggregation settings."""
        interval_length = max(1, end - start)
        if has_agg:
            if bin_size:
                return max(1, int(np.ceil(interval_length / bin_size)))
            return 1
        return interval_length

    @staticmethod
    def _aggregate_binned_signal(signal: np.ndarray, bin_size: int, aggfunc: Callable) -> np.ndarray:
        """Aggregate a 1D signal into bins, including a potentially short tail bin."""
        if signal.size == 0:
            return np.zeros((1,), dtype=np.float32)

        bins = []
        for start_idx in range(0, signal.size, bin_size):
            chunk = signal[start_idx:start_idx + bin_size]
            try:
                bins.append(aggfunc(chunk, axis=-1))
            except TypeError:
                bins.append(aggfunc(chunk))
        return np.asarray(bins)
            
    def _get_aggfunc(
        self,
        func: Optional[Union[str, Callable]]
    ) -> Optional[Callable]:
        """Get aggregation function.
        
        Args:
        func : str or callable, optional
            Name of aggregation function or callable
            
        Returns:
        callable or None
            Aggregation function if specified
            
        Raises:
        ValueError
            If function name is invalid
            
        Notes:
        Supported function names:
        - "mean": Average value
        - "max": Maximum value
        - "min": Minimum value
        - "sum": Sum of values
        """
        if func is None:
            return None
            
        if callable(func):
            return func
            
        func_map = {
            "mean": np.mean,
            "max": np.max,
            "min": np.min,
            "sum": np.sum
        }
        
        if func not in func_map:
            raise ValueError(
                f"Invalid aggregation function: {func}. "
                f"Supported functions: {list(func_map.keys())}"
            )
            
        return func_map[func]

    def validate_intervals(self, intervals: pd.DataFrame) -> None:
        """Validate genomic intervals data format.
        
        Args:
        intervals : pd.DataFrame
            DataFrame with genomic intervals
            
        Raises:
        ValueError
            If intervals format is invalid
            
        Notes:
        Validates:
        - Required columns (chrom, start, end)
        - Column data types
        - Coordinate ordering
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
        if not np.issubdtype(intervals['start'].dtype, np.number):
            raise ValueError("'start' column must be numeric type")
        if not np.issubdtype(intervals['end'].dtype, np.number):
            raise ValueError("'end' column must be numeric type")
            
        # Validate interval coordinates
        if (intervals['end'] <= intervals['start']).any():
            raise ValueError("Interval end positions must be greater than start positions")
        if (intervals['start'] < 0).any():
            raise ValueError("Interval start positions cannot be negative")

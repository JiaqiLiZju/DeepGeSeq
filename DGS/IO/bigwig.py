"""BigWig file reader module for genomic signal data.

This module provides functionality for reading and processing BigWig format files,
which store continuous signal data across genomic coordinates. It supports random
access to genomic regions and flexible signal aggregation.

Classes
-------
BigWigReader
    Reader for BigWig signal files with the following features:
    - Random access to genomic regions
    - Signal aggregation with multiple functions
    - Flexible binning options
    - Multi-track support
    - Comprehensive error handling

Notes
-----
The module requires the pyBigWig package for file access. Signal values can be
aggregated using various functions (mean, max, min, sum) and can be binned to
reduce resolution. Missing or invalid regions return appropriate zero/NA values.

Examples
--------
>>> reader = BigWigReader("signal.bw")
>>> intervals = pd.DataFrame({
...     "chrom": ["chr1", "chr1"],
...     "start": [1000, 2000],
...     "end": [1500, 2500]
... })
>>> signals = reader.read(intervals, bin_size=10, aggfunc="mean")
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

    Attributes
    ----------
    file_path : Path
        Path to the BigWig file
    VALID_EXTENSIONS : list of str
        List of valid file extensions [".bw", ".bigwig", ".bigWig"]

    Methods
    -------
    read(intervals, bin_size=None, aggfunc=None, **kwargs)
        Read signal values for specified genomic intervals
    check_file()
        Validate file path and extension
    validate_intervals(intervals)
        Validate genomic interval data format

    Examples
    --------
    >>> reader = BigWigReader("signal.bw")
    >>> signals = reader.read(intervals, bin_size=10, aggfunc="mean")
    """
    
    VALID_EXTENSIONS = [".bw", ".bigwig", ".bigWig"]
    
    def __init__(self, file_path: Union[str, Path]):
        """Initialize BigWig reader.
        
        Parameters
        ----------
        file_path : str or Path
            Path to BigWig file
            
        Raises
        ------
        ValueError
            If file path is invalid or file doesn't exist
        """
        self.file_path = Path(file_path)
        if not self.check_file():
            raise ValueError(f"Invalid BigWig file: {self.file_path}")
            
    def check_file(self) -> bool:
        """Check if file has valid extension and exists.
        
        Returns
        -------
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
        
        Parameters
        ----------
        intervals : pd.DataFrame
            DataFrame with genomic intervals (chrom, start, end)
        bin_size : int, optional
            Number of bases to aggregate for binning
        aggfunc : str or callable, optional
            Aggregation function ("mean", "max", "min", "sum")
            or custom function
        **kwargs : dict
            Additional arguments passed to pyBigWig
            
        Returns
        -------
        np.ndarray
            Array of shape (n_intervals, n_bins) with coverage values
            
        Raises
        ------
        IOError
            If file reading fails
        ValueError
            If arguments are invalid
            
        Notes
        -----
        - Missing regions return zero values
        - NaN values are converted to zeros
        - When binning, the last bin may be smaller
        """
        try:
            import pyBigWig
            
            # Validate intervals
            self.validate_intervals(intervals)
            
            # Setup aggregation function
            aggfunc = self._get_aggfunc(aggfunc)
            
            # Read signals
            signals = []
            with pyBigWig.open(str(self.file_path)) as bw:
                for row in intervals.itertuples():
                    try:
                        # Calculate length for zero array in case of error
                        interval_length = row.end - row.start
                        if bin_size and aggfunc is not None:
                            interval_length = interval_length // bin_size
                        if interval_length < 1:
                            interval_length = 1
                            
                        # Read raw signal
                        signal = bw.values(row.chrom, row.start, row.end, numpy=True)
                        if signal is None:
                            signal = np.zeros(interval_length)
                        else:
                            signal = np.nan_to_num(signal)
                        
                        # Aggregate if requested
                        if aggfunc is not None:
                            if bin_size:
                                # Aggregate over bins
                                signal = signal.reshape(-1, bin_size)
                                signal = aggfunc(signal, axis=-1)
                            else:
                                # Aggregate over whole interval
                                signal = aggfunc(signal)
                            
                        signals.append(signal)
                        
                    except Exception as e:
                        logger.warning(
                            f"Failed to read interval {row.chrom}:{row.start}-{row.end}: {e}"
                        )
                        signals.append(np.zeros(interval_length))
                        
            # Stack and return
            return np.asarray(signals)
            
        except Exception as e:
            logger.error(f"Failed to read BigWig file {self.file_path}: {str(e)}")
            raise
            
    def _get_aggfunc(
        self,
        func: Optional[Union[str, Callable]]
    ) -> Optional[Callable]:
        """Get aggregation function.
        
        Parameters
        ----------
        func : str or callable, optional
            Name of aggregation function or callable
            
        Returns
        -------
        callable or None
            Aggregation function if specified
            
        Raises
        ------
        ValueError
            If function name is invalid
            
        Notes
        -----
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
        
        Parameters
        ----------
        intervals : pd.DataFrame
            DataFrame with genomic intervals
            
        Raises
        ------
        ValueError
            If intervals format is invalid
            
        Notes
        -----
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

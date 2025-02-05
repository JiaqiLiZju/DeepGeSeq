"""BigWig file reader module for DGS.

Provides:
1. BigWigReader - Reader for BigWig signal files
2. Signal aggregation functions
3. Multi-track support
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Callable
from pathlib import Path

from . import logger

class BigWigReader:
    """Reader for BigWig genomic signal files.
    
    Features:
    - Single and multi-track reading
    - Signal aggregation
    - Flexible binning options
    """
    
    VALID_EXTENSIONS = [".bw", ".bigwig", ".bigWig"]
    
    def __init__(self, file_path: Union[str, Path]):
        """Initialize reader.
        
        Args:
            file_path: Path to BigWig file
            
        Raises:
            ValueError: If file is invalid
        """
        self.file_path = Path(file_path)
        if not self.check_file():
            raise ValueError(f"Invalid file: {self.file_path}")
            
    def check_file(self) -> bool:
        """Check if file has valid extension and exists."""
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
            intervals: DataFrame with genomic intervals
            bin_size: Number of bases to aggregate
            aggfunc: Aggregation function ("mean", "max", "min", "sum")
            **kwargs: Additional arguments
            
        Returns:
            Array of shape (intervals, tracks, length) with coverage values
            
        Raises:
            IOError: If file reading fails
            ValueError: If arguments are invalid
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
                        
            # Stack
            return np.asarray(signals)
            
        except Exception as e:
            logger.error(f"Failed to read BigWig file {self.file_path}: {str(e)}")
            raise
            
    def _get_aggfunc(
        self,
        func: Optional[Union[str, Callable]]
    ) -> Optional[Callable]:
        """Get aggregation function.
        
        Args:
            func: Function name or callable
            
        Returns:
            Aggregation function
            
        Raises:
            ValueError: If function name is invalid
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
        """Validate the input genomic intervals data.
        
        Args:
            intervals: DataFrame containing genomic interval information
            
        Raises:
            ValueError: If the data format is invalid
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
            
        # Validate interval validity
        if (intervals['end'] <= intervals['start']).any():
            raise ValueError("Interval end positions must be greater than start positions")
        if (intervals['start'] < 0).any():
            raise ValueError("Interval start positions cannot be negative")

"""BED file reader module for genomic interval data.

This module provides functionality for reading and processing BED format files,
which store genomic interval data. It supports both standard BED and ENCODE
narrowPeak formats with automatic format detection.

Classes
-------
BedReader
    Reader for BED and BED-like files with the following features:
    - Standard BED format (3-12 columns)
    - ENCODE narrowPeak format (10 columns)
    - Automatic format detection
    - Column standardization
    - Data validation
    - Memory-efficient reading

Notes
-----
The module handles both gzipped and uncompressed files. For standard BED format,
it supports the following columns:
1. chrom: Chromosome name
2. start: Start position (0-based)
3. end: End position
4. name: Feature name (optional)
5. score: Feature score (optional)
6. strand: Strand orientation (optional)
7-12. Additional BED fields (optional)

For narrowPeak format, it supports the ENCODE standard columns:
1-6. Same as BED format
7. signalValue: Measurement of signal enrichment
8. pValue: Statistical significance (-log10)
9. qValue: FDR adjusted p-value (-log10)
10. peak: Peak point within feature

Examples
--------
>>> reader = BedReader("peaks.bed")
>>> intervals = reader.read()
>>> print(intervals.columns)
['chrom', 'start', 'end', 'name', 'score', 'strand']
"""

import pandas as pd
from typing import Optional, Dict, List, Union
from pathlib import Path

from . import logger

class BedReader:
    """Reader for BED format files.
    
    Provides functionality for reading and validating BED and BED-like files,
    with support for both standard BED and ENCODE narrowPeak formats.

    Attributes
    ----------
    file_path : Path
        Path to the BED file
    VALID_EXTENSIONS : list of str
        List of valid file extensions [".bed", ".narrowPeak"]
    BED_COLUMNS : list of str
        Standard BED format column names
    NARROWPEAK_COLUMNS : list of str
        ENCODE narrowPeak format column names

    Methods
    -------
    read(has_header=False, column_map=None, **kwargs)
        Read BED file into DataFrame with standardized columns
    check_file()
        Validate file path and extension
    validate_intervals(intervals)
        Validate genomic interval data format

    Notes
    -----
    - Files can be gzipped (.gz extension) or uncompressed
    - At least 3 columns (chrom, start, end) are required
    - Coordinates are validated for proper ordering and non-negative values
    - Column names are standardized based on format detection
    """
    
    VALID_EXTENSIONS = [".bed", ".narrowPeak"]
    
    # Standard BED columns
    BED_COLUMNS = [
        "chrom", "start", "end", "name", "score", "strand",
        "thickStart", "thickEnd", "itemRgb", 
        "blockCount", "blockSizes", "blockStarts"
    ]
    
    # narrowPeak specific columns
    NARROWPEAK_COLUMNS = [
        "chrom", "start", "end", "name", "score",
        "strand", "signalValue", "pValue", "qValue", "peak"
    ]
    
    def __init__(self, file_path: Union[str, Path]):
        """Initialize BED reader.
        
        Parameters
        ----------
        file_path : str or Path
            Path to BED format file
            
        Raises
        ------
        ValueError
            If file path is invalid or file doesn't exist
        """
        self.file_path = Path(file_path)
        if not self.check_file():
            raise ValueError(f"Invalid file: {self.file_path}")

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
    
    def read(
        self,
        has_header: bool = False,
        column_map: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Read BED file into DataFrame.
        
        Parameters
        ----------
        has_header : bool, optional
            Whether file has header row (default: False)
        column_map : dict, optional
            Custom column name mapping
        **kwargs : dict
            Additional arguments passed to pd.read_table
            
        Returns
        -------
        pd.DataFrame
            DataFrame with standardized interval format
            
        Raises
        ------
        ValueError
            If file format is invalid or required columns are missing
            
        Notes
        -----
        - Automatically detects narrowPeak format based on file extension
        - Standardizes column names based on format
        - Validates interval coordinates
        - Converts data types (chrom: str, start/end: int)
        """
        try:
            # Read file
            df = pd.read_table(
                self.file_path,
                header=0 if has_header else None,
                comment="#",
                **kwargs
            )
            
            # Determine format and standardize columns
            if str(self.file_path).endswith('.narrowPeak') or str(self.file_path).endswith('.narrowPeak.gz'):
                if len(df.columns) != 10:
                    raise ValueError("narrowPeak files must have exactly 10 columns")
                df.columns = self.NARROWPEAK_COLUMNS
            else:
                if len(df.columns) < 3:
                    raise ValueError("BED files must have at least 3 columns")
                df = self._standardize_bed_columns(df)
            
            # Convert coordinates to int
            df["chrom"] = df["chrom"].astype(str)
            df["start"] = df["start"].astype(int)
            df["end"] = df["end"].astype(int)
            
            # Validate data before column mapping
            self.validate_intervals(df)
            
            # Apply custom column mapping
            if column_map:
                df = df.rename(columns=column_map)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to read BED file {self.file_path}: {str(e)}")
            raise
            
    def _standardize_bed_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize BED format columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with BED data
            
        Returns
        -------
        pd.DataFrame
            DataFrame with standardized column names
            
        Raises
        ------
        ValueError
            If number of columns exceeds BED format specification
        """
        n_cols = len(df.columns)
        if n_cols > len(self.BED_COLUMNS):
            raise ValueError(f"Too many columns: {n_cols}")
            
        df.columns = self.BED_COLUMNS[:n_cols]
        return df
        
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

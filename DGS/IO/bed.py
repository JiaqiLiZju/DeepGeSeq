"""BED file reader module for DGS.

This module provides:
1. BedReader - Reader for BED and BED-like files (BED, narrowPeak)
2. Standardized interval data loading
3. Flexible column handling and validation
"""

import pandas as pd
from typing import Optional, Dict, List, Union
from pathlib import Path

from . import logger

class BedReader:
    """Reader for BED format files.
    
    Features:
    - Standard BED format (3-12 columns)
    - ENCODE narrowPeak format
    - Automatic column naming
    - Data validation
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
        """Initialize reader.
        
        Args:
            file_path: Path to BED file
            
        Raises:
            ValueError: If file is invalid
        """
        self.file_path = Path(file_path)
        if not self.check_file():
            raise ValueError(f"Invalid file: {self.file_path}")

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
    
    def read(
        self,
        has_header: bool = False,
        column_map: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Read BED file into DataFrame.
        
        Args:
            has_header: Whether file has header row
            column_map: Custom column name mapping
            **kwargs: Additional arguments for pd.read_table
            
        Returns:
            DataFrame with standardized interval format
            
        Raises:
            ValueError: If file format is invalid
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
        """Standardize BED format columns."""
        n_cols = len(df.columns)
        if n_cols > len(self.BED_COLUMNS):
            raise ValueError(f"Too many columns: {n_cols}")
            
        df.columns = self.BED_COLUMNS[:n_cols]
        return df
        
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

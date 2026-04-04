"""Interval operations and interval container classes.

Purpose:
    Provide overlap, distance, merge, and statistics utilities for genomic ranges.

Main Responsibilities:
    - Compute overlaps and nearest intervals between interval sets.
    - Merge interval ranges and summarize interval-level statistics.
    - Expose `Interval` and `NamedInterval` classes for pipeline integration.

Key Runtime Notes:
    - Overlap outputs include stable `query_index` and `target_index` fields.
    - Interval coordinates follow half-open semantics (`start` inclusive, `end` exclusive).
    - BED readers are used when interval objects are created from file paths.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple, Callable
from pathlib import Path
import logging

from ..IO import BedReader

logger = logging.getLogger("dgs.interval")

def find_overlaps(
    query_intervals: pd.DataFrame,
    target_intervals: pd.DataFrame,
    chrom_col: str = 'chrom',
    start_col: str = 'start',
    end_col: str = 'end'
) -> pd.DataFrame:
    """Find overlapping regions between two sets of genomic intervals.
    
    This function:
    - Processes intervals chromosome by chromosome
    - Calculates overlap coordinates
    - Computes overlap lengths
    - Preserves original interval information
    
    Args:
        query_intervals: Query intervals DataFrame
        target_intervals: Target intervals DataFrame
        chrom_col: Name of chromosome column
        start_col: Name of start position column
        end_col: Name of end position column
        
    Returns:
        pd.DataFrame: Overlapping intervals with columns:
            - Original interval columns
            - overlap_start: Start of overlap region
            - overlap_end: End of overlap region
            - overlap_length: Length of overlap
            - query_index: Original index from `query_intervals`
            - target_index: Original index from `target_intervals`
            - query_* columns: Original query interval data
            
    Example:
        >>> query = pd.DataFrame({
        ...     'chrom': ['chr1'], 
        ...     'start': [100], 
        ...     'end': [200]
        ... })
        >>> target = pd.DataFrame({
        ...     'chrom': ['chr1'], 
        ...     'start': [150], 
        ...     'end': [250]
        ... })
        >>> overlaps = find_overlaps(query, target)
    """
    results = []
    core_cols = {chrom_col, start_col, end_col}
    query_extra_cols = [col for col in query_intervals.columns if col not in core_cols]

    # Iterate only over chromosomes present in both sets.
    common_chroms = np.intersect1d(
        query_intervals[chrom_col].unique(),
        target_intervals[chrom_col].unique(),
    )

    for chrom in common_chroms:
        query_chrom = query_intervals[query_intervals[chrom_col] == chrom]
        target_chrom = target_intervals[target_intervals[chrom_col] == chrom]
        if query_chrom.empty or target_chrom.empty:
            continue

        # Sort by start once, then scan with monotonic pointers.
        query_sorted = query_chrom.sort_values(start_col, kind="mergesort")
        target_sorted = target_chrom.sort_values(start_col, kind="mergesort")

        q_starts = query_sorted[start_col].to_numpy()
        q_ends = query_sorted[end_col].to_numpy()
        q_indices = query_sorted.index.to_numpy()

        t_starts = target_sorted[start_col].to_numpy()
        t_ends = target_sorted[end_col].to_numpy()
        n_target = len(target_sorted)
        left_ptr = 0
        chrom_target_positions = []
        chrom_query_indices = []
        chrom_overlap_starts = []
        chrom_overlap_ends = []
        chrom_query_payload = {f"query_{col}": [] for col in query_extra_cols}

        for q_pos, (q_start, q_end, q_index) in enumerate(zip(q_starts, q_ends, q_indices)):
            while left_ptr < n_target and t_ends[left_ptr] <= q_start:
                left_ptr += 1
            if left_ptr >= n_target:
                break

            right_ptr = np.searchsorted(t_starts, q_end, side="left")
            if right_ptr <= left_ptr:
                continue

            active_mask = t_ends[left_ptr:right_ptr] > q_start
            if not np.any(active_mask):
                continue

            active_positions = np.nonzero(active_mask)[0] + left_ptr
            chrom_target_positions.extend(active_positions.tolist())
            chrom_query_indices.extend([q_index] * len(active_positions))

            overlap_start = np.maximum(q_start, t_starts[active_positions])
            overlap_end = np.minimum(q_end, t_ends[active_positions])
            chrom_overlap_starts.extend(overlap_start.tolist())
            chrom_overlap_ends.extend(overlap_end.tolist())

            if query_extra_cols:
                query_row = query_sorted.iloc[q_pos]
                for col in query_extra_cols:
                    chrom_query_payload[f"query_{col}"].extend([query_row[col]] * len(active_positions))

        if chrom_target_positions:
            overlap_regions = target_sorted.iloc[chrom_target_positions].copy()
            overlap_regions["overlap_start"] = np.asarray(chrom_overlap_starts)
            overlap_regions["overlap_end"] = np.asarray(chrom_overlap_ends)
            overlap_regions["overlap_length"] = (
                overlap_regions["overlap_end"].to_numpy() - overlap_regions["overlap_start"].to_numpy()
            )
            overlap_regions["query_index"] = np.asarray(chrom_query_indices)
            overlap_regions["target_index"] = overlap_regions.index.to_numpy()
            for col_name, values in chrom_query_payload.items():
                overlap_regions[col_name] = values
            results.append(overlap_regions)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)

def merge_intervals(
    intervals: pd.DataFrame,
    min_distance: int = 0,
    chrom_col: str = 'chrom',
    start_col: str = 'start',
    end_col: str = 'end'
) -> pd.DataFrame:
    """Merge overlapping or nearby genomic intervals.
    
    This function:
    - Processes intervals by chromosome
    - Merges overlapping regions
    - Handles nearby intervals within min_distance
    - Preserves chromosome information
    
    Args:
        intervals: Input intervals DataFrame
        min_distance: Maximum distance between intervals to merge
        chrom_col: Name of chromosome column
        start_col: Name of start position column
        end_col: Name of end position column
        
    Returns:
        pd.DataFrame: Merged intervals with columns:
            - chrom: Chromosome name
            - start: Start position
            - end: End position
            
    Example:
        >>> df = pd.DataFrame({
        ...     'chrom': ['chr1', 'chr1'],
        ...     'start': [100, 150],
        ...     'end': [200, 250]
        ... })
        >>> merged = merge_intervals(df, min_distance=10)
    """
    merged_intervals = []
    
    # Process each chromosome
    for chrom in intervals[chrom_col].unique():
        # Get sorted intervals for current chromosome
        chrom_intervals = intervals[intervals[chrom_col] == chrom].sort_values(start_col)
        
        if len(chrom_intervals) == 0:
            continue
            
        # Initialize with first interval
        current_start = chrom_intervals.iloc[0][start_col]
        current_end = chrom_intervals.iloc[0][end_col]
        
        # Process remaining intervals
        for _, interval in chrom_intervals.iloc[1:].iterrows():
            if interval[start_col] <= current_end + min_distance:
                # Merge intervals
                current_end = max(current_end, interval[end_col])
            else:
                # Add current merged interval and start new one
                merged_intervals.append({
                    chrom_col: chrom,
                    start_col: current_start,
                    end_col: current_end
                })
                current_start = interval[start_col]
                current_end = interval[end_col]
                
        # Add last interval
        merged_intervals.append({
            chrom_col: chrom,
            start_col: current_start,
            end_col: current_end
        })
        
    return pd.DataFrame(merged_intervals)

def find_closest(
    query_intervals: pd.DataFrame,
    target_intervals: pd.DataFrame,
    k: int = 1,
    max_distance: Optional[int] = None,
    chrom_col: str = 'chrom',
    start_col: str = 'start',
    end_col: str = 'end'
) -> pd.DataFrame:
    """Find k closest intervals for each query interval.
    
    This function:
    - Processes intervals by chromosome
    - Calculates distances between intervals
    - Finds k nearest neighbors
    - Optionally filters by maximum distance
    
    Args:
        query_intervals: Query intervals DataFrame
        target_intervals: Target intervals DataFrame
        k: Number of closest intervals to find
        max_distance: Maximum distance to consider
        chrom_col: Name of chromosome column
        start_col: Name of start position column
        end_col: Name of end position column
        
    Returns:
        pd.DataFrame: Closest intervals with columns:
            - Original interval columns
            - distance: Distance to query interval
            - query_* columns: Original query interval data
            
    Example:
        >>> query = pd.DataFrame({
        ...     'chrom': ['chr1'],
        ...     'start': [1000],
        ...     'end': [2000]
        ... })
        >>> target = pd.DataFrame({
        ...     'chrom': ['chr1'],
        ...     'start': [2500, 3000],
        ...     'end': [3500, 4000]
        ... })
        >>> closest = find_closest(query, target, k=1)
    """
    results = []
    
    # Process each chromosome
    for chrom in query_intervals[chrom_col].unique():
        # Get intervals for current chromosome
        query_chrom = query_intervals[query_intervals[chrom_col] == chrom]
        target_chrom = target_intervals[target_intervals[chrom_col] == chrom]
        
        if len(query_chrom) == 0 or len(target_chrom) == 0:
            continue
            
        # Calculate distances for each interval
        for idx, interval in query_chrom.iterrows():
            # Calculate distances to start and end
            start_distances = np.abs(target_chrom[start_col] - interval[end_col])
            end_distances = np.abs(target_chrom[end_col] - interval[start_col])
            
            # Get minimum distance for each interval
            distances = np.minimum(start_distances, end_distances)
            
            # Filter by maximum distance if specified
            if max_distance is not None:
                valid_mask = distances <= max_distance
                if not valid_mask.any():
                    continue
                distances = distances[valid_mask]
                target_chrom_filtered = target_chrom[valid_mask]
            else:
                target_chrom_filtered = target_chrom
                
            # Get k closest intervals
            closest_indices = np.argsort(distances)[:k]
            closest_intervals = target_chrom_filtered.iloc[closest_indices].copy()
            closest_intervals['distance'] = distances[closest_indices]
            closest_intervals['query_index'] = idx  # Add query index
            
            # Add query interval information
            for col in interval.index:
                if col not in [chrom_col, start_col, end_col]:
                    closest_intervals[f"query_{col}"] = interval[col]
                    
            results.append(closest_intervals)
            
    if not results:
        return pd.DataFrame()
        
    return pd.concat(results, ignore_index=True)

def get_interval_stats(
    intervals: pd.DataFrame,
    chrom_col: str = 'chrom',
    start_col: str = 'start',
    end_col: str = 'end'
) -> pd.Series:
    """Calculate comprehensive statistics for genomic intervals.
    
    This function computes:
    - Basic count statistics
    - Length distribution metrics
    - Chromosome coverage
    - Interval characteristics
    
    Args:
        intervals: Input intervals DataFrame
        chrom_col: Name of chromosome column
        start_col: Name of start position column
        end_col: Name of end position column
        
    Returns:
        pd.Series: Statistics including:
            - total_intervals: Number of intervals
            - total_bases: Total bases covered
            - mean_length: Average interval length
            - median_length: Median interval length
            - min_length: Minimum interval length
            - max_length: Maximum interval length
            - unique_chromosomes: Number of unique chromosomes
            
    Example:
        >>> df = pd.DataFrame({
        ...     'chrom': ['chr1', 'chr2'],
        ...     'start': [100, 200],
        ...     'end': [200, 400]
        ... })
        >>> stats = get_interval_stats(df)
    """
    lengths = intervals[end_col] - intervals[start_col]
    return pd.Series({
        'total_intervals': len(intervals),
        'total_bases': lengths.sum(),
        'mean_length': lengths.mean(),
        'median_length': lengths.median(),
        'min_length': lengths.min(),
        'max_length': lengths.max(),
        'unique_chromosomes': intervals[chrom_col].nunique()
    })

class Interval:
    """Class for managing genomic intervals.
    
    This class provides:
    - Interval validation and processing
    - Overlap and distance calculations
    - Statistical analysis
    - File format conversion
    
    Attributes:
        data: DataFrame containing interval information
        chrom_col: Name of chromosome column
        start_col: Name of start position column
        end_col: Name of end position column
        stats: Basic interval statistics
        
    Example:
        >>> intervals = Interval("regions.bed")
        >>> intervals.merge_overlapping()
        >>> intervals.to_bed("merged.bed")
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path],
        chrom_col: str = 'chrom',
        start_col: str = 'start',
        end_col: str = 'end',
        **kwargs
    ):
        """Initialize Interval object.
        
        Args:
            data: Input data (DataFrame or file path)
            chrom_col: Name of chromosome column
            start_col: Name of start position column
            end_col: Name of end position column
            **kwargs: Additional arguments for file reading
            
        Raises:
            ValueError: If required columns are missing
            ValueError: If intervals are invalid
        """
        self.chrom_col = chrom_col
        self.start_col = start_col
        self.end_col = end_col
        
        # Load data
        if isinstance(data, (str, Path)):
            reader = BedReader(data)
            self.data = reader.read(**kwargs)
            # Map column names if different from BED standard
            if chrom_col != 'chrom':
                self.data = self.data.rename(columns={'chrom': chrom_col})
            if start_col != 'start':
                self.data = self.data.rename(columns={'start': start_col})
            if end_col != 'end':
                self.data = self.data.rename(columns={'end': end_col})
        else:
            self.data = data.copy()
            
        # Validate columns
        self._validate_columns()
        
        # Calculate basic statistics
        self.stats = get_interval_stats(
            self.data,
            self.chrom_col,
            self.start_col,
            self.end_col
        )
        
    def _validate_columns(self) -> None:
        """Validate required columns and interval format.
        
        This function checks:
        - Required columns exist
        - Coordinates are valid
        - Start positions are non-negative
        
        Raises:
            ValueError: If validation fails
        """
        # Check required columns
        required = [self.chrom_col, self.start_col, self.end_col]
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        # Validate coordinates
        if (self.data[self.end_col] <= self.data[self.start_col]).any():
            raise ValueError("Found intervals with end <= start")
            
        if (self.data[self.start_col] < 0).any():
            raise ValueError("Found negative start positions")
    
    @classmethod
    def from_bed(
        cls,
        file_path: Union[str, Path],
        chrom_col: str = 'chrom',
        start_col: str = 'start',
        end_col: str = 'end',
        **kwargs
    ) -> 'Interval':
        """Create Interval object from BED file.
        
        Args:
            file_path: Path to BED file
            chrom_col: Name of chromosome column
            start_col: Name of start position column
            end_col: Name of end position column
            **kwargs: Additional arguments for BedReader
            
        Returns:
            Interval: New interval object
            
        Example:
            >>> intervals = Interval.from_bed("regions.bed")
        """
        reader = BedReader(file_path)
        data = reader.read(**kwargs)
        
        # Rename columns if different from BED standard
        if chrom_col != 'chrom':
            data = data.rename(columns={'chrom': chrom_col})
        if start_col != 'start':
            data = data.rename(columns={'start': start_col})
        if end_col != 'end':
            data = data.rename(columns={'end': end_col})
        
        return cls(data, chrom_col, start_col, end_col)
            
    def merge_with(
        self,
        other: Union[pd.DataFrame, 'Interval', str, Path],
        how: str = 'inner',
        min_overlap: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Merge with another set of intervals.
        
        This function:
        - Finds overlapping regions
        - Applies merge strategy
        - Handles non-overlapping intervals
        - Preserves interval attributes
        
        Args:
            other: Other intervals (DataFrame, Interval, or file)
            how: Merge type ('inner', 'left', 'right', 'outer')
            min_overlap: Minimum overlap required
            **kwargs: Additional arguments for file reading
            
        Returns:
            pd.DataFrame: Merged intervals
            
        Example:
            >>> intervals1 = Interval("regions1.bed")
            >>> intervals2 = Interval("regions2.bed")
            >>> merged = intervals1.merge_with(intervals2, how='inner')
        """
        # Get other intervals
        if isinstance(other, Interval):
            other_df = other.data
        elif isinstance(other, pd.DataFrame):
            other_df = other
        else:
            other_df = pd.read_csv(other, **kwargs)
            
        # Find overlaps
        overlaps = find_overlaps(
            self.data,
            other_df,
            self.chrom_col,
            self.start_col,
            self.end_col
        )
        
        # Filter by minimum overlap if specified
        if min_overlap is not None:
            overlaps = overlaps[overlaps['overlap_length'] >= min_overlap]

        if overlaps.empty:
            if how == 'inner':
                return overlaps
            if how == 'left':
                return self.data.copy()
            if how == 'right':
                return other_df.copy()
            if how == 'outer':
                return pd.concat([self.data.copy(), other_df.copy()], ignore_index=True, sort=False)
            raise ValueError(f"Invalid merge type: {how}")
            
        # Handle merge type
        if how == 'inner':
            return overlaps
        elif how == 'left':
            # Add non-overlapping intervals from self
            used_indices = set(overlaps['query_index'].unique())
            remaining = self.data.loc[~self.data.index.isin(used_indices)]
            if len(remaining) > 0:
                return pd.concat([overlaps, remaining], ignore_index=True, sort=False)
            return overlaps
        elif how == 'right':
            # Add non-overlapping intervals from other
            used_indices = set(overlaps['target_index'].unique())
            remaining = other_df.loc[~other_df.index.isin(used_indices)]
            if len(remaining) > 0:
                return pd.concat([overlaps, remaining], ignore_index=True, sort=False)
            return overlaps
        elif how == 'outer':
            # Add non-overlapping intervals from both
            used_self = set(overlaps['query_index'].unique())
            used_other = set(overlaps['target_index'].unique())
            remaining_self = self.data.loc[~self.data.index.isin(used_self)]
            remaining_other = other_df.loc[~other_df.index.isin(used_other)]
            return pd.concat([overlaps, remaining_self, remaining_other], ignore_index=True, sort=False)
        else:
            raise ValueError(f"Invalid merge type: {how}")
            
    def merge_overlapping(self, min_distance: int = 0) -> 'Interval':
        """Merge overlapping or nearby intervals.
        
        Args:
            min_distance: Maximum distance between intervals to merge
            
        Returns:
            Interval: New interval object with merged intervals
            
        Example:
            >>> intervals = Interval("regions.bed")
            >>> merged = intervals.merge_overlapping(min_distance=100)
        """
        merged = merge_intervals(
            self.data,
            min_distance,
            self.chrom_col,
            self.start_col,
            self.end_col
        )
        return Interval(merged)
        
    def find_closest(
        self,
        other: Union[pd.DataFrame, 'Interval', str, Path],
        k: int = 1,
        max_distance: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Find k closest intervals.
        
        Args:
            other: Other intervals to search
            k: Number of closest intervals to find
            max_distance: Maximum distance to consider
            **kwargs: Additional arguments for file reading
            
        Returns:
            pd.DataFrame: Closest intervals with distances
            
        Example:
            >>> intervals = Interval("regions.bed")
            >>> closest = intervals.find_closest("other.bed", k=2)
        """
        # Get other intervals
        if isinstance(other, Interval):
            other_df = other.data
        elif isinstance(other, pd.DataFrame):
            other_df = other
        else:
            other_df = pd.read_csv(other, **kwargs)
            
        return find_closest(
            self.data,
            other_df,
            k,
            max_distance,
            self.chrom_col,
            self.start_col,
            self.end_col
        )
        
    def get_stats(self) -> pd.Series:
        """Get interval statistics.
        
        Returns:
            pd.Series: Comprehensive interval statistics
        """
        return self.stats
        
    def to_bed(self, output_file: Union[str, Path], **kwargs) -> None:
        """Save intervals to BED file.
        
        Args:
            output_file: Path to save BED file
            **kwargs: Additional arguments for DataFrame.to_csv()
            
        Example:
            >>> intervals = Interval("regions.bed")
            >>> intervals.to_bed("processed.bed")
        """
        # Map column names back to BED standard if needed
        data = self.data.copy()
        if self.chrom_col != 'chrom':
            data = data.rename(columns={self.chrom_col: 'chrom'})
        if self.start_col != 'start':
            data = data.rename(columns={self.start_col: 'start'})
        if self.end_col != 'end':
            data = data.rename(columns={self.end_col: 'end'})
            
        # Ensure BED format order
        bed_columns = ['chrom', 'start', 'end']
        other_columns = [col for col in data.columns if col not in bed_columns]
        data = data[bed_columns + other_columns]
        
        # Write to file
        data.to_csv(output_file, sep='\t', index=False, **kwargs)
        
    def copy(self) -> 'Interval':
        """Create a deep copy.
        
        Returns:
            Interval: New interval object with copied data
        """
        return Interval(
            self.data.copy(),
            self.chrom_col,
            self.start_col,
            self.end_col
        )

    def find_overlaps(
        self,
        query_intervals: pd.DataFrame,
        target_intervals: pd.DataFrame
    ) -> pd.DataFrame:
        """Find overlapping intervals between two sets of intervals.
        
        Args:
            query_intervals: Query intervals DataFrame
            target_intervals: Target intervals DataFrame
            
        Returns:
            pd.DataFrame: Overlapping intervals
        """
        return find_overlaps(
            query_intervals,
            target_intervals,
            self.chrom_col,
            self.start_col,
            self.end_col
        )

class NamedInterval(Interval):
    """Class for managing genomic intervals with unique identifiers.
    
    This class extends Interval with:
    - Unique name handling
    - Name validation
    - Automatic name generation
    - Name preservation in operations
    
    Attributes:
        name_col: Name of identifier column
        
    Example:
        >>> intervals = NamedInterval("regions.bed", name_col="gene_id")
        >>> intervals.merge_overlapping(keep_names=True)
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path],
        chrom_col: str = 'chrom',
        start_col: str = 'start',
        end_col: str = 'end',
        name_col: str = 'name',
        name_prefix: str = 'interval_',
        **kwargs
    ):
        """Initialize NamedInterval object.
        
        Args:
            data: Input data (DataFrame or file path)
            chrom_col: Name of chromosome column
            start_col: Name of start position column
            end_col: Name of end position column
            name_col: Name of identifier column
            name_prefix: Prefix for auto-generated names
            **kwargs: Additional arguments for file reading
        """
        super().__init__(data, chrom_col, start_col, end_col, **kwargs)
        self.name_col = name_col
        
        # Handle name column from BED file
        if isinstance(data, (str, Path)) and 'name' in self.data.columns and name_col != 'name':
            self.data = self.data.rename(columns={'name': name_col})
            
        # Ensure name column exists
        if self.name_col not in self.data.columns:
            self.data[self.name_col] = [
                f"{name_prefix}{i}" for i in range(len(self.data))
            ]
        else:
            # Fill missing names
            missing_mask = self.data[self.name_col].isna()
            if missing_mask.any():
                missing_count = missing_mask.sum()
                self.data.loc[missing_mask, self.name_col] = [
                    f"{name_prefix}{i}" for i in range(missing_count)
                ]
            
            # Ensure unique names
            if self.data[self.name_col].duplicated().any():
                logger.warning("Found duplicate names, appending unique suffixes")
                # Create a mapping of duplicate names to their count
                name_counts = self.data[self.name_col].value_counts()
                duplicates = name_counts[name_counts > 1].index
                
                # Add suffix only to duplicates
                for name in duplicates:
                    mask = self.data[self.name_col] == name
                    dups = self.data[mask]
                    for i, idx in enumerate(dups.index):
                        if i > 0:  # Skip the first occurrence
                            self.data.loc[idx, self.name_col] = f"{name}_{i}"
                            
    def merge_with(
        self,
        other: Union[pd.DataFrame, 'Interval', str, Path],
        how: str = 'inner',
        min_overlap: Optional[int] = None,
        suffixes: Tuple[str, str] = ('_x', '_y'),
        **kwargs
    ) -> pd.DataFrame:
        """Merge with another set of intervals.
        
        Args:
            other: Other intervals to merge with
            how: Merge type ('inner', 'left', 'right', 'outer')
            min_overlap: Minimum overlap required
            suffixes: Suffixes for duplicate column names
            **kwargs: Additional arguments for file reading
            
        Returns:
            pd.DataFrame: Merged intervals with preserved names
        """
        merged = super().merge_with(other, how, min_overlap, **kwargs)
        
        # Add name information
        if isinstance(other, NamedInterval):
            other_df = other.data
            if self.name_col in other_df.columns:
                merged[f"{self.name_col}{suffixes[1]}"] = merged.index.map(
                    other_df[self.name_col]
                )
                
        return merged
        
    def merge_overlapping(self, min_distance: int = 0, keep_names: bool = True) -> 'NamedInterval':
        """Merge overlapping or nearby intervals.
        
        Args:
            min_distance: Maximum distance between intervals to merge
            keep_names: Whether to preserve original names
            
        Returns:
            NamedInterval: New interval object with merged intervals
        """
        merged = super().merge_overlapping(min_distance)
        
        if keep_names:
            # Group original names by merged intervals
            name_groups = []
            for _, merged_interval in merged.data.iterrows():
                overlaps = (
                    (self.data[self.chrom_col] == merged_interval[self.chrom_col]) &
                    (self.data[self.start_col] >= merged_interval[self.start_col]) &
                    (self.data[self.end_col] <= merged_interval[self.end_col])
                )
                name_groups.append(self.data.loc[overlaps, self.name_col].tolist())
            merged.data[f"original_{self.name_col}s"] = name_groups
            
        return NamedInterval(merged.data)
        
    def find_closest(
        self,
        other: Union[pd.DataFrame, 'Interval', str, Path],
        k: int = 1,
        max_distance: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Find k closest intervals.
        
        Args:
            other: Other intervals to search
            k: Number of closest intervals to find
            max_distance: Maximum distance to consider
            **kwargs: Additional arguments for file reading
            
        Returns:
            pd.DataFrame: Closest intervals with preserved names
        """
        closest = super().find_closest(other, k, max_distance, **kwargs)
        
        # Add name information
        closest[f"query_{self.name_col}"] = closest['query_index'].map(
            self.data[self.name_col]
        )
        
        if isinstance(other, NamedInterval):
            other_df = other.data
            if self.name_col in other_df.columns:
                closest[f"target_{self.name_col}"] = closest.index.map(
                    other_df[self.name_col]
                )
                
        return closest
        
    def copy(self) -> 'NamedInterval':
        """Create a deep copy.
        
        Returns:
            NamedInterval: New interval object with copied data
        """
        return NamedInterval(
            self.data.copy(),
            self.chrom_col,
            self.start_col,
            self.end_col,
            self.name_col
        )

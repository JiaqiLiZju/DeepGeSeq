"""Tests for Target class and helper functions."""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import os
try:
    import pyBigWig
except ImportError:  # pragma: no cover - optional dependency
    pyBigWig = None

from DGS.Data.Target import Target, get_class_distribution, get_imbalance_metrics, get_rare_label_stats

def create_test_bigwig(file_path: Path, intervals: pd.DataFrame) -> None:
    """Create a test BigWig file with base-level signals.
    
    Args:
        file_path: Path to output BigWig file
        intervals: DataFrame with intervals to generate signals for
    """
    # Get chromosome sizes from intervals
    chrom_sizes = {
        chrom: int(df['end'].max()) + 100
        for chrom, df in intervals.groupby('chrom')
    }
    
    # Create BigWig file
    bw = pyBigWig.open(str(file_path), 'w')
    bw.addHeader(list(chrom_sizes.items()))
    
    # Add test data with known patterns
    for chrom in chrom_sizes:
        size = chrom_sizes[chrom]
        starts = list(range(0, size))
        ends = [x + 1 for x in starts]
        
        # Create signal pattern:
        # - Background noise (0-0.5)
        # - Peaks at specific positions (height 2-5)
        values = np.random.uniform(0, 0.5, size)
        
        # Add peaks near interval centers
        interval_mask = intervals['chrom'] == chrom
        for _, row in intervals[interval_mask].iterrows():
            center = (row['start'] + row['end']) // 2
            width = 20
            peak_start = max(0, center - width)
            peak_end = min(size, center + width)
            x = np.arange(peak_start, peak_end)
            peak = 5 * np.exp(-0.5 * ((x - center) / 10)**2)
            values[peak_start:peak_end] = np.maximum(values[peak_start:peak_end], peak)
        
        bw.addEntries(
            chroms=[chrom] * len(starts),
            starts=starts,
            ends=ends,
            values=values.tolist()
        )
    
    bw.close()

@unittest.skipUnless(pyBigWig is not None, "pyBigWig is required for Target tests")
class TestTarget(unittest.TestCase):
    """Test Target class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create base intervals
        self.intervals = pd.DataFrame({
            'chrom': ['chr1', 'chr1', 'chr2'],
            'start': [100, 300, 500],
            'end':   [200, 400, 600]
        })
        
        # Create test files
        self.bigwig_file = self.test_dir / "signal.bw"
        create_test_bigwig(self.bigwig_file, self.intervals)
        
        # Create test BED file
        self.bed_file = self.test_dir / "test.bed"
        with open(self.bed_file, 'w') as f:
            f.write("chr1\t100\t200\n")
            f.write("chr1\t300\t400\n")
            f.write("chr2\t500\t600\n")
        
        # Define basic BigWig task
        self.bigwig_task = {
            'task_name': 'signal',
            'file_path': str(self.bigwig_file),
            'file_type': 'bigwig',
            'task_type': 'regression',
            'bin_size': None,  # Base-level resolution
            'aggfunc': 'mean'
        }
        
        self.bed_task = {
            'task_name': 'bed_task',
            'file_path': str(self.bed_file),
            'file_type': 'bed',
            'task_type': 'binary',
            'target_column': 'name'  # Assuming the BED file has a 'name' column
        }
        
        self.tasks = [
            self.bigwig_task,
            {
                'task_name': 'test_task',
                'file_path': str(self.bigwig_file),
                'file_type': 'bigwig',
                'task_type': 'binary',
                'bin_size': None,
                'aggfunc': 'mean',
                'threshold': 0.5
            },
            self.bed_task
        ]
        
        self.target = Target(self.intervals, self.tasks)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_get_labels(self):
        """Test getting labels from the target."""
        labels = self.target.get_labels('test_task')
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(labels.shape[0], len(self.intervals))
    
    def test_get_stats(self):
        """Test getting statistics from the target."""
        stats = self.target.get_stats('test_task')
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
    
    def test_get_task_info(self):
        """Test getting task information."""
        task_info = self.target.get_task_info()
        self.assertIsInstance(task_info, pd.DataFrame)
        self.assertIn('test_task', task_info.index)
    
    def test_get_intervals(self):
        """Test getting intervals."""
        intervals = self.target.get_intervals()
        self.assertIsInstance(intervals, pd.DataFrame)
        self.assertEqual(intervals.shape[0], len(self.intervals))
    
    def test_bigwig_signal_shape(self):
        """Test shape of BigWig signal."""
        values = self.target.get_labels('signal')
        self.assertEqual(values.shape[0], len(self.intervals))
    
    def test_bigwig_signal_range(self):
        """Test range of BigWig signal values."""
        values = self.target.get_labels('signal')
        self.assertTrue(np.all(values >= 0))
        self.assertTrue(np.all(values <= 5))  # Assuming max peak height is 5
    
    def test_bigwig_with_threshold(self):
        """Test BigWig signal thresholding."""
        self.bigwig_task['threshold'] = 2.0
        target = Target(self.intervals, [self.bigwig_task])
        
        values = target.get_labels('signal')
        self.assertTrue(np.all(np.isin(values, [0, 1])))
    
    def test_bigwig_statistics(self):
        """Test BigWig signal statistics."""
        target = Target(self.intervals, [self.bigwig_task])
        stats = target.get_stats('signal')
        
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
    
    def test_bigwig_return_format(self):
        """Test return format of BigWig signal."""
        target = Target(self.intervals, [self.bigwig_task])
        
        values = target.get_labels('signal')
        self.assertTrue(isinstance(values, np.ndarray))
        self.assertEqual(len(values), len(self.intervals))
    
    def test_bigwig_with_bin_size(self):
        """Test BigWig signal with bin_size > 1."""
        bigwig_task_with_bins = {
            'task_name': 'signal_with_bins',
            'file_path': str(self.bigwig_file),
            'file_type': 'bigwig',
            'task_type': 'regression',
            'bin_size': 5,
            'aggfunc': 'mean'
        }
        
        target = Target(self.intervals, [bigwig_task_with_bins])
        values = target.get_labels('signal_with_bins')
        print(values.shape)
        
        # Check dimensions
        self.assertEqual(len(values), len(self.intervals))
        expected_bins = (self.intervals['end'] - self.intervals['start']) // 5
        for i, bins in enumerate(expected_bins):
            self.assertEqual(values[i].shape[0], bins)  # Ensure the shape matches expected bins
    
    def test_bigwig_return_format_with_bins(self):
        """Test return format with bin_size > 1."""
        bigwig_task_with_bins = {
            'task_name': 'signal_with_bins',
            'file_path': str(self.bigwig_file),
            'file_type': 'bigwig',
            'task_type': 'regression',
            'bin_size': 5,
            'aggfunc': 'mean'
        }
        
        target = Target(self.intervals, [bigwig_task_with_bins])
        values = target.get_labels('signal_with_bins')
        
        self.assertTrue(isinstance(values, np.ndarray))
        self.assertEqual(len(values), len(self.intervals))
        
        expected_bins = (self.intervals['end'] - self.intervals['start']) // 5
        for i, bins in enumerate(expected_bins):
            self.assertEqual(values[i].shape[0], bins)

    def test_bed_labels(self):
        """Test getting labels from the BED task."""
        labels = self.target.get_labels('bed_task')
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(labels.shape[0], len(self.intervals))  # Check if the number of labels matches intervals

    def test_bed_statistics(self):
        """Test getting statistics from the BED task."""
        stats = self.target.get_stats('bed_task')
        self.assertIn('positive_count', stats)
        self.assertIn('negative_count', stats)

    def test_bed_task_info(self):
        """Test getting task information for the BED task."""
        task_info = self.target.get_task_info()
        self.assertIsInstance(task_info, pd.DataFrame)
        self.assertIn('bed_task', task_info.index)

def usage_example():
    """Example usage of Target class with BigWig data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        
        # Create intervals
        intervals = pd.DataFrame({
            'chrom': ['chr1', 'chr1', 'chr2'],
            'start': [100, 300, 500],
            'end':   [200, 400, 600]
        })
        
        # Create test BigWig file
        bigwig_file = tmp_dir / "signal.bw"
        create_test_bigwig(bigwig_file, intervals)
        
        # Define tasks
        tasks = [
            {
                'task_name': 'signal_base',
                'file_path': str(bigwig_file),
                'file_type': 'bigwig',
                'bin_size': None,
                'aggfunc': 'mean'
            },
            {
                'task_name': 'signal_binary',
                'file_path': str(bigwig_file),
                'file_type': 'bigwig',
                'bin_size': None,
                'threshold': 2.0
            }
        ]
        
        # Create Target object
        target = Target(intervals, tasks)
        
        # Get signals for a single task
        signals = target.get_labels('signal_base')
        print("\nSignal array shape:", signals.shape)
        
        # Get all tasks
        all_signals = target.get_labels()
        print("\nAll tasks:", list(all_signals.keys()))
        for name, data in all_signals.items():
            print(f"{name} shape:", data.shape)

if __name__ == '__main__':
    print("Running usage example...")
    usage_example()
    
    print("\nRunning tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 

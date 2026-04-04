"""Tests for BigWig file reader module."""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import pyBigWig
except ImportError:  # pragma: no cover - optional dependency
    pyBigWig = None

from DGS.IO.bigwig import BigWigReader

def usage_example():
    """Example usage of BigWigReader class.
    
    This example demonstrates:
    1. Basic signal reading
    2. Signal aggregation
    3. Multi-interval processing
    4. Error handling
    """
    # Create a sample BigWig file
    with tempfile.TemporaryDirectory() as tmp_dir:
        bw_path = Path(tmp_dir) / "example.bw"
        
        # Create test BigWig
        bw = pyBigWig.open(str(bw_path), 'w')
        # Add a chromosome
        bw.addHeader([("chr1", 1000), ("chr2", 1000)])
        # Add some test data
        chroms = ["chr1"] * 100 + ["chr2"] * 100
        starts = list(range(0, 900, 10)) + list(range(0, 900, 10))
        ends = [s + 10 for s in starts]
        values = [float(i % 5) for i in range(200)]
        bw.addEntries(chroms, starts, ends=ends, values=values)
        bw.close()
        
        # Initialize reader
        reader = BigWigReader(bw_path)
        
        # Example 1: Basic signal reading
        intervals = pd.DataFrame({
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [100, 200]
        })
        
        signals = reader.read(intervals)
        print("Basic signal reading:")
        print("Shape:", signals.shape)
        print("Values:", signals[0, 0, :10])  # First interval, first 10 values
        
        # Example 2: Signal aggregation with binning
        signals = reader.read(intervals, bin_size=10, aggfunc="mean")
        print("\nAggregated signals (bin_size=10):")
        print("Shape:", signals.shape)
        print("Values:", signals[0, 0, :])
        
        # Example 3: Different aggregation functions
        for func in ["mean", "max", "min", "sum"]:
            signals = reader.read(intervals, aggfunc=func)
            print(f"\n{func.capitalize()} aggregation:")
            print("Values:", signals[0, 0, :])
        
        # Example 4: Custom aggregation function
        def custom_agg(x, **kwargs):
            return np.percentile(x, 75, **kwargs)
        
        signals = reader.read(intervals, aggfunc=custom_agg)
        print("\nCustom aggregation (75th percentile):")
        print("Values:", signals[0, 0, :])

@unittest.skipUnless(pyBigWig is not None, "pyBigWig is required for BigWig tests")
class TestBigWigReader(unittest.TestCase):
    """Test BigWigReader functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test BigWig file
        self.bw_file = Path(self.test_dir) / "test.bw"
        bw = pyBigWig.open(str(self.bw_file), 'w')
        
        # Add header with chromosome sizes
        bw.addHeader([("chr1", 1000), ("chr2", 1000)])
        
        # Create test data
        chroms = []
        starts = []
        ends = []
        values = []
        
        # Add data for chr1
        for i in range(0, 900, 10):
            chroms.append("chr1")
            starts.append(i)
            ends.append(i + 10)
            values.append(float(i % 5))
            
        # Add data for chr2
        for i in range(0, 900, 10):
            chroms.append("chr2")
            starts.append(i)
            ends.append(i + 10)
            values.append(float(i % 5))
        
        # Add entries to the bigwig file
        bw.addEntries(chroms, starts, ends=ends, values=values)
        bw.close()
        
        # Create test intervals
        self.test_intervals = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [0, 100, 200],
            "end": [50, 150, 250]
        })
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
        
    def test_initialization(self):
        """Test BigWigReader initialization."""
        # Test valid initialization
        reader = BigWigReader(self.bw_file)
        self.assertTrue(self.bw_file.exists())
        
        # Test with invalid file
        with self.assertRaises(ValueError):
            BigWigReader(Path(self.test_dir) / "nonexistent.bw")
            
        # Test with invalid extension
        invalid_file = Path(self.test_dir) / "test.txt"
        invalid_file.write_text("some content")
        with self.assertRaises(ValueError):
            BigWigReader(invalid_file)
            
    def test_basic_reading(self):
        """Test basic signal reading."""
        reader = BigWigReader(self.bw_file)
        signals = reader.read(self.test_intervals)
        
        # Check shape
        self.assertEqual(len(signals.shape), 2)  # (intervals, length)
        self.assertEqual(signals.shape[0], len(self.test_intervals))
        
        # Check values
        self.assertTrue(np.all(signals >= 0))
        self.assertTrue(np.all(signals <= 4))  # Max value in test data
        
    def test_aggregation(self):
        """Test signal aggregation."""
        reader = BigWigReader(self.bw_file)
        
        # Test different aggregation functions
        for func in ["mean", "max", "min", "sum"]:
            signals = reader.read(self.test_intervals, aggfunc=func)
            self.assertIsNotNone(signals)
            
        # Test custom function
        def custom_agg(x, **kwargs):
            return np.percentile(x, 75, **kwargs)
            
        signals = reader.read(self.test_intervals, aggfunc=custom_agg)
        self.assertIsNotNone(signals)
        
        # Test invalid function
        with self.assertRaises(ValueError):
            reader.read(self.test_intervals, aggfunc="invalid")
            
    def test_binning(self):
        """Test signal binning."""
        reader = BigWigReader(self.bw_file)
        
        # Test different bin sizes
        bin_sizes = [2, 5, 10]
        for bin_size in bin_sizes:
            signals = reader.read(self.test_intervals, bin_size=bin_size, aggfunc="mean")
            # Check that output length is correct
            for i, row in self.test_intervals.iterrows():
                expected_length = (row.end - row.start) // bin_size
                self.assertEqual(signals[i].shape[0], expected_length)
                
    def test_validation(self):
        """Test interval validation."""
        reader = BigWigReader(self.bw_file)
        
        # Test invalid chromosome
        invalid_chrom = pd.DataFrame({
            "chrom": ["chr3"],  # Non-existent chromosome
            "start": [0],
            "end": [100]
        })
        signals = reader.read(invalid_chrom)
        self.assertTrue(np.all(signals == 0))  # Should return zeros for invalid regions
        
        # Test invalid coordinates
        invalid_coords = pd.DataFrame({
            "chrom": ["chr1"],
            "start": [-100],  # Negative start
            "end": [100]
        })
        with self.assertRaises(ValueError):
            reader.read(invalid_coords)
            
        invalid_coords = pd.DataFrame({
            "chrom": ["chr1"],
            "start": [100],
            "end": [50]  # end < start
        })
        with self.assertRaises(ValueError):
            reader.read(invalid_coords)
            
    def test_error_handling(self):
        """Test error handling."""
        reader = BigWigReader(self.bw_file)
        
        # Test missing required columns
        invalid_df = pd.DataFrame({
            "chromosome": ["chr1"],  # Wrong column name
            "start": [0],
            "end": [100]
        })
        with self.assertRaises(ValueError):
            reader.read(invalid_df)
            
        # Test invalid data types
        invalid_types = pd.DataFrame({
            "chrom": [1],  # Should be string
            "start": ["0"],  # Should be numeric
            "end": [100]
        })
        with self.assertRaises(ValueError):
            reader.read(invalid_types)

if __name__ == '__main__':
    # Run the example if called directly
    usage_example()
    unittest.main() 

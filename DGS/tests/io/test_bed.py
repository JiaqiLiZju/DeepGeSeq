"""Tests for BED file reader module."""

import unittest
import tempfile
import shutil
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

from DGS.IO.bed import BedReader

HAS_BGZIP = shutil.which("bgzip") is not None

def usage_example():
    """Example usage of BedReader class.
    
    This example demonstrates:
    1. Basic BED file reading
    2. Working with different BED formats (BED3, BED6, narrowPeak)
    3. Custom column mapping
    4. Data validation
    """
    # Create a sample BED file
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Example 1: Basic BED3 format
        bed3_path = Path(tmp_dir) / "example.bed"
        bed3_content = (
            "chr1\t100\t200\n"
            "chr1\t500\t600\n"
            "chr2\t1000\t2000\n"
        )
        bed3_path.write_text(bed3_content)
        
        reader = BedReader(bed3_path)
        df = reader.read()
        print("Basic BED3 format:")
        print(df)
        print("\nColumns:", df.columns.tolist())
        
        # Example 2: BED6 format with header
        bed6_path = Path(tmp_dir) / "example_bed6.bed"
        bed6_content = (
            "chrom\tstart\tend\tname\tscore\tstrand\n"
            "chr1\t100\t200\tpeak1\t1.23\t+\n"
            "chr2\t500\t600\tpeak2\t4.56\t-\n"
        )
        bed6_path.write_text(bed6_content)
        
        reader = BedReader(bed6_path)
        df = reader.read(has_header=True)
        print("\nBED6 format with header:")
        print(df)
        
        # Example 3: narrowPeak format
        narrowpeak_path = Path(tmp_dir) / "example.narrowPeak"
        narrowpeak_content = (
            "chr1\t100\t200\tpeak1\t1000\t.\t123.45\t-1\t-1\t50\n"
            "chr2\t500\t600\tpeak2\t2000\t.\t234.56\t-1\t-1\t75\n"
        )
        narrowpeak_path.write_text(narrowpeak_content)
        
        reader = BedReader(narrowpeak_path)
        df = reader.read()
        print("\nnarrowPeak format:")
        print(df)
        
        # Example 4: Custom column mapping
        column_map = {
            "chrom": "chromosome",
            "start": "start_pos",
            "end": "end_pos"
        }
        df = reader.read(column_map=column_map)
        print("\nCustom column mapping:")
        print(df)

class TestBedReader(unittest.TestCase):
    """Test BedReader functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test BED3 content
        self.bed3_content = (
            "chr1\t100\t200\n"
            "chr2\t300\t400\n"
        )
        
        # Create test BED6 content
        self.bed6_content = (
            "chr1\t100\t200\tpeak1\t1.23\t+\n"
            "chr2\t300\t400\tpeak2\t4.56\t-\n"
        )
        
        # Create test narrowPeak content
        self.narrowpeak_content = (
            "chr1\t100\t200\tpeak1\t1000\t.\t123.45\t-1\t-1\t50\n"
            "chr2\t300\t400\tpeak2\t2000\t.\t234.56\t-1\t-1\t75\n"
        )
        
        # Create test files
        self.bed3_file = Path(self.test_dir) / "test.bed"
        self.bed3_file.write_text(self.bed3_content)
        
        self.bed6_file = Path(self.test_dir) / "test_bed6.bed"
        self.bed6_file.write_text(self.bed6_content)
        
        self.narrowpeak_file = Path(self.test_dir) / "test.narrowPeak"
        self.narrowpeak_file.write_text(self.narrowpeak_content)
        
        # Create compressed file when bgzip exists in PATH
        self.bed3_gz = None
        if HAS_BGZIP:
            self.bed3_gz = Path(self.test_dir) / "test.bed.gz"
            with open(self.bed3_file, 'rb') as f_in:
                with open(self.bed3_gz, 'wb') as f_out:
                    subprocess.run(['bgzip', '-c'], 
                                 stdin=f_in,
                                 stdout=f_out,
                                 check=True)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
        
    def test_initialization(self):
        """Test BedReader initialization."""
        # Test valid initialization
        reader = BedReader(self.bed3_file)
        self.assertTrue(self.bed3_file.exists())
        
        # Test with compressed file when available
        if self.bed3_gz is not None:
            reader = BedReader(self.bed3_gz)
            self.assertTrue(self.bed3_gz.exists())
        
        # Test with invalid file
        with self.assertRaises(ValueError):
            BedReader(Path(self.test_dir) / "nonexistent.bed")
            
        # Test with invalid extension
        invalid_file = Path(self.test_dir) / "test.txt"
        invalid_file.write_text("some content")
        with self.assertRaises(ValueError):
            BedReader(invalid_file)
            
    def test_read_bed3(self):
        """Test reading BED3 format."""
        reader = BedReader(self.bed3_file)
        df = reader.read()
        
        # Check basic properties
        self.assertEqual(len(df), 2)
        self.assertEqual(len(df.columns), 3)
        self.assertTrue(all(col in df.columns for col in ["chrom", "start", "end"]))
        
        # Check data types
        self.assertTrue(df["chrom"].dtype == object)
        self.assertTrue(pd.api.types.is_integer_dtype(df["start"]))
        self.assertTrue(pd.api.types.is_integer_dtype(df["end"]))
        
        # Check values
        self.assertEqual(df.iloc[0]["chrom"], "chr1")
        self.assertEqual(df.iloc[0]["start"], 100)
        self.assertEqual(df.iloc[0]["end"], 200)
        
    def test_read_bed6(self):
        """Test reading BED6 format."""
        reader = BedReader(self.bed6_file)
        df = reader.read()
        
        # Check columns
        self.assertEqual(len(df.columns), 6)
        expected_cols = ["chrom", "start", "end", "name", "score", "strand"]
        self.assertTrue(all(col in df.columns for col in expected_cols))
        
        # Check values
        self.assertEqual(df.iloc[0]["name"], "peak1")
        self.assertEqual(df.iloc[0]["strand"], "+")
        
    def test_read_narrowpeak(self):
        """Test reading narrowPeak format."""
        reader = BedReader(self.narrowpeak_file)
        df = reader.read()
        
        # Check columns
        self.assertEqual(len(df.columns), 10)
        self.assertTrue(all(col in df.columns for col in reader.NARROWPEAK_COLUMNS))
        
        # Check values
        self.assertEqual(df.iloc[0]["signalValue"], 123.45)
        self.assertEqual(df.iloc[0]["peak"], 50)
        
    def test_column_mapping(self):
        """Test custom column mapping."""
        reader = BedReader(self.bed3_file)
        column_map = {
            "chrom": "chromosome",
            "start": "start_pos",
            "end": "end_pos"
        }
        df = reader.read(column_map=column_map)
        
        # Check mapped columns
        self.assertTrue(all(col in df.columns for col in column_map.values()))
        
    def test_validation(self):
        """Test interval validation."""
        # Create invalid intervals
        invalid_start = Path(self.test_dir) / "invalid_start.bed"
        invalid_start.write_text("chr1\t-100\t200\n")
        
        invalid_end = Path(self.test_dir) / "invalid_end.bed"
        invalid_end.write_text("chr1\t200\t100\n")
        
        # Test negative start
        reader = BedReader(invalid_start)
        with self.assertRaises(ValueError) as cm:
            reader.read()
        self.assertIn("cannot be negative", str(cm.exception))
        
        # Test end <= start
        reader = BedReader(invalid_end)
        with self.assertRaises(ValueError) as cm:
            reader.read()
        self.assertIn("must be greater than start", str(cm.exception))
        
    @unittest.skipUnless(HAS_BGZIP, "bgzip is required for compressed BED tests")
    def test_compressed_file(self):
        """Test reading compressed BED file."""
        reader = BedReader(self.bed3_gz)
        df = reader.read()
        
        # Should read same content as uncompressed
        reader_uncompressed = BedReader(self.bed3_file)
        df_uncompressed = reader_uncompressed.read()
        
        pd.testing.assert_frame_equal(df, df_uncompressed)

if __name__ == '__main__':
    # Run the example if called directly
    usage_example()
    unittest.main() 

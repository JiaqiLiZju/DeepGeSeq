"""Tests for IO module."""

import unittest
import tempfile
import shutil
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
try:
    import pysam
except ImportError:  # pragma: no cover - optional dependency
    pysam = None
try:
    import pyBigWig
except ImportError:  # pragma: no cover - optional dependency
    pyBigWig = None

from DGS.IO import FastaReader, BedReader, BigWigReader, logger

HAS_BGZIP = shutil.which("bgzip") is not None

@unittest.skipUnless(
    pysam is not None and pyBigWig is not None,
    "pysam and pyBigWig are required for IO tests",
)
class TestIOModule(unittest.TestCase):
    """Test IO module functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test files
        self.create_test_files()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
        
    def create_test_files(self):
        """Create test files for different formats."""
        # Create FASTA file
        self.fasta_file = Path(self.test_dir) / "test.fasta"
        fasta_content = (
            ">chr1\n"
            "ACTGACTGACTG\n"
            ">chr2\n"
            "GCATGCATGCAT\n"
        )
        self.fasta_file.write_text(fasta_content)
        pysam.faidx(str(self.fasta_file))
        
        # Create BED file
        self.bed_file = Path(self.test_dir) / "test.bed"
        bed_content = (
            "chr1\t0\t5\tpeak1\t1.23\t+\n"
            "chr2\t5\t10\tpeak2\t4.56\t-\n"
        )
        self.bed_file.write_text(bed_content)
        
        # Create BigWig file
        self.bw_file = Path(self.test_dir) / "test.bw"
        bw = pyBigWig.open(str(self.bw_file), 'w')
        bw.addHeader([("chr1", 100), ("chr2", 100)])
        
        # Add test data (must be sorted by chromosome and start position)
        chroms = []
        starts = []
        ends = []
        values = []
        
        # Add chr1 data
        for i in range(0, 50, 10):
            chroms.append("chr1")
            starts.append(i)
            ends.append(i + 10)
            values.append(float(i/10))
            
        # Add chr2 data
        for i in range(0, 50, 10):
            chroms.append("chr2")
            starts.append(i)
            ends.append(i + 10)
            values.append(float(i/10))
            
        # Add entries to bigwig file
        bw.addEntries(chroms, starts, ends=ends, values=values)
        bw.close()
        
    def test_imports(self):
        """Test that all readers can be imported."""
        self.assertTrue(hasattr(FastaReader, '__init__'))
        self.assertTrue(hasattr(BedReader, '__init__'))
        self.assertTrue(hasattr(BigWigReader, '__init__'))
        
    def test_logger(self):
        """Test that logger is properly configured."""
        self.assertTrue(hasattr(logger, 'error'))
        self.assertTrue(hasattr(logger, 'warning'))
        self.assertTrue(hasattr(logger, 'info'))
        
    def test_file_validation(self):
        """Test file validation across readers."""
        # Test valid files
        FastaReader(self.fasta_file)
        BedReader(self.bed_file)
        BigWigReader(self.bw_file)
        
        # Test invalid files
        invalid_file = Path(self.test_dir) / "invalid.txt"
        invalid_file.write_text("some content")
        
        with self.assertRaises(ValueError):
            FastaReader(invalid_file)
        with self.assertRaises(ValueError):
            BedReader(invalid_file)
        with self.assertRaises(ValueError):
            BigWigReader(invalid_file)
            
        # Test non-existent files
        with self.assertRaises(ValueError):
            FastaReader(Path(self.test_dir) / "nonexistent.fa")
        with self.assertRaises(ValueError):
            BedReader(Path(self.test_dir) / "nonexistent.bed")
        with self.assertRaises(ValueError):
            BigWigReader(Path(self.test_dir) / "nonexistent.bw")
            
    def test_interval_validation(self):
        """Test interval validation across readers."""
        # Create test intervals
        valid_intervals = pd.DataFrame({
            "chrom": ["chr1", "chr2"],
            "start": [0, 5],
            "end": [5, 10]
        })
        
        invalid_intervals = pd.DataFrame({
            "chrom": ["chr1"],
            "start": [-1],  # Invalid start
            "end": [10]
        })
        
        # Test with FASTA reader
        with FastaReader(self.fasta_file) as reader:
            reader.validate_intervals(valid_intervals)
            with self.assertRaises(ValueError):
                reader.validate_intervals(invalid_intervals)
                
        # Test with BED reader
        reader = BedReader(self.bed_file)
        reader.validate_intervals(valid_intervals)
        with self.assertRaises(ValueError):
            reader.validate_intervals(invalid_intervals)
            
        # Test with BigWig reader
        reader = BigWigReader(self.bw_file)
        reader.validate_intervals(valid_intervals)
        with self.assertRaises(ValueError):
            reader.validate_intervals(invalid_intervals)
            
    @unittest.skipUnless(HAS_BGZIP, "bgzip is required for compressed-file IO tests")
    def test_compressed_files(self):
        """Test handling of compressed files."""
        # Create compressed files
        fasta_gz = Path(self.test_dir) / "test.fasta.gz"
        bed_gz = Path(self.test_dir) / "test.bed.gz"
        
        # Compress files using bgzip
        with open(self.fasta_file, 'rb') as f_in:
            with open(fasta_gz, 'wb') as f_out:
                subprocess.run(['bgzip', '-c'], stdin=f_in, stdout=f_out, check=True)
                
        with open(self.bed_file, 'rb') as f_in:
            with open(bed_gz, 'wb') as f_out:
                subprocess.run(['bgzip', '-c'], stdin=f_in, stdout=f_out, check=True)
        
        # Test readers with compressed files
        FastaReader(fasta_gz)
        BedReader(bed_gz)
        
    def test_common_interface(self):
        """Test common interface across readers."""
        # All readers should have these methods
        readers = [
            FastaReader(self.fasta_file),
            BedReader(self.bed_file),
            BigWigReader(self.bw_file)
        ]
        
        for reader in readers:
            self.assertTrue(hasattr(reader, 'check_file'))
            self.assertTrue(hasattr(reader, 'validate_intervals'))
            self.assertTrue(hasattr(reader, 'read'))

if __name__ == '__main__':
    unittest.main() 

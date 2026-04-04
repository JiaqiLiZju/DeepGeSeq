"""Tests for FASTA file reader module."""

import unittest
import tempfile
import subprocess
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
try:
    import pysam
except ImportError:  # pragma: no cover - optional dependency
    pysam = None

from DGS.IO.fasta import FastaReader

HAS_BGZIP = shutil.which("bgzip") is not None

def usage_example():
    """Example usage of FastaReader class.
    
    This example demonstrates:
    1. Basic initialization and file handling
    2. Reading specific sequences by coordinates
    3. Batch processing of multiple regions
    4. Working with compressed FASTA files
    """
    # Create a sample FASTA file
    with tempfile.TemporaryDirectory() as tmp_dir:
        fasta_path = Path(tmp_dir) / "example.fasta"
        
        # Write sample content
        fasta_content = (
            ">chr1\n"
            "ACTGACTGACTGACTGACTG\n"
            ">chr2\n"
            "GCATGCATGCATGCATGCAT\n"
        )
        fasta_path.write_text(fasta_content)
        
        # Create index
        pysam.faidx(str(fasta_path))
        
        # Initialize reader
        reader = FastaReader(fasta_path)
        
        # Example 1: Get all reference sequences
        with reader:
            print("Available references:", reader.references)
            print("Sequence lengths:", reader.lengths)
        
        # Example 2: Read specific intervals
        intervals = pd.DataFrame({
            "chrom": ["chr1", "chr2"],
            "start": [0, 5],
            "end": [10, 15]
        })
        
        with reader:
            sequences = reader.read(intervals)
            print("\nExtracted sequences:")
            for i, seq in enumerate(sequences):
                print(f"Interval {i+1}: {seq}")
        
        # Example 3: Batch processing
        large_intervals = pd.concat([intervals] * 4)  # Create more intervals
        
        with reader:
            print("\nProcessing in batches:")
            for i, batch in enumerate(reader.read(large_intervals, batch_size=2)):
                print(f"Batch {i+1}:", batch)
        
        # Example 4: Direct sequence access
        with reader:
            seq = reader.get_sequence("chr1", 5, 15)
            print("\nDirect access:", seq)

# Test cases below
@unittest.skipUnless(pysam is not None, "pysam is required for FASTA tests")
class TestFastaReader(unittest.TestCase):
    """Test FastaReader functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test FASTA content
        self.fasta_content = (
            ">chr1\n"
            "ACTGACTGACTGACTGACTG\n"  # 20 bases
            "TGACTGACTGACTGACTGAC\n"  # 40 bases
            ">chr2\n"
            "GCATGCATGCATGCATGCAT\n"  # 20 bases
            "ATGCATGCATGCATGCATGC\n"  # 40 bases
        )
        
        # Create test files
        self.fasta_file = Path(self.test_dir) / "test.fasta"
        self.fasta_file.write_text(self.fasta_content)
        
        # Create bgzipped file when bgzip exists in PATH
        self.fasta_gz = None
        if HAS_BGZIP:
            self.fasta_gz = Path(self.test_dir) / "test.fasta.gz"
            with open(self.fasta_file, 'rb') as f_in:
                with open(self.fasta_gz, 'wb') as f_out:
                    subprocess.run(['bgzip', '-c'], 
                                 stdin=f_in,
                                 stdout=f_out,
                                 check=True)
            
        # Create index files
        pysam.faidx(str(self.fasta_file))
        if self.fasta_gz is not None:
            pysam.faidx(str(self.fasta_gz))
        
        # Create test intervals
        self.test_intervals = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [0, 10, 5],
            "end": [10, 20, 15]
        })
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
        
    def test_initialization(self):
        """Test FastaReader initialization."""
        # Test valid initialization
        reader = FastaReader(self.fasta_file)
        self.assertTrue(Path(str(self.fasta_file) + '.fai').exists())
        
        # Test with bgzipped file when available
        if self.fasta_gz is not None:
            reader = FastaReader(self.fasta_gz)
            self.assertTrue(Path(str(self.fasta_gz) + '.fai').exists())
        
        # Test with invalid file
        with self.assertRaises(ValueError):
            FastaReader(Path(self.test_dir) / "nonexistent.fasta")
            
        # Test with invalid extension
        invalid_file = Path(self.test_dir) / "test.txt"
        invalid_file.write_text("some content")
        with self.assertRaises(ValueError):
            FastaReader(invalid_file)
        
    def test_references(self):
        """Test getting reference sequences."""
        reader = FastaReader(self.fasta_file)
        
        with reader:
            refs = reader.references
            self.assertEqual(len(refs), 2)
            self.assertIn("chr1", refs)
            self.assertIn("chr2", refs)
            
    def test_lengths(self):
        """Test getting reference lengths."""
        reader = FastaReader(self.fasta_file)
        
        with reader:
            lengths = reader.lengths
            self.assertEqual(len(lengths), 2)
            self.assertEqual(lengths["chr1"], 40)
            self.assertEqual(lengths["chr2"], 40)
            
    def test_read_intervals(self):
        """Test reading sequences from intervals."""
        reader = FastaReader(self.fasta_file)
        
        with reader:
            # Test reading specific intervals
            sequences = reader.read(self.test_intervals)
            self.assertEqual(len(sequences), 3)
            
            # Verify sequence lengths
            for seq, (_, row) in zip(sequences, self.test_intervals.iterrows()):
                self.assertEqual(len(seq), row.end - row.start)
                
            # Verify sequence content
            self.assertEqual(sequences[0], "ACTGACTGAC")  # chr1:0-10
            self.assertEqual(sequences[1], "TGACTGACTG")  # chr1:10-20
            self.assertEqual(sequences[2], "CATGCATGCA")  # chr2:5-15
        
    @unittest.skipUnless(HAS_BGZIP, "bgzip is required for compressed FASTA tests")
    def test_read_compressed(self):
        """Test reading from bgzipped FASTA."""
        reader = FastaReader(self.fasta_gz)
        
        with reader:
            sequences = reader.read(self.test_intervals)
            
            self.assertEqual(len(sequences), 3)
            self.assertEqual(sequences[0], "ACTGACTGAC")
            self.assertEqual(sequences[1], "TGACTGACTG")
            self.assertEqual(sequences[2], "CATGCATGCA")
        
    def test_sequence_validation(self):
        """Test sequence validation."""
        reader = FastaReader(self.fasta_file)
        
        with reader:
            # Test invalid chromosome
            invalid_chrom = pd.DataFrame({
                "chrom": ["chr3"],  # Non-existent chromosome
                "start": [0],
                "end": [10]
            })
            with self.assertRaises(ValueError) as cm:
                reader.get_sequence("chr3", 0, 10)
            self.assertIn("Unknown chromosome", str(cm.exception))
            
            # Test invalid coordinates
            with self.assertRaises(ValueError) as cm:
                reader.get_sequence("chr1", -10, 10)  # Negative start
            self.assertIn("Negative start position", str(cm.exception))
            
            with self.assertRaises(ValueError) as cm:
                reader.get_sequence("chr1", 10, 5)  # end < start
            self.assertIn("Invalid interval", str(cm.exception))
        
    def test_batch_processing(self):
        """Test batch sequence extraction."""
        reader = FastaReader(self.fasta_file)
        
        with reader:
            # Create larger test data
            large_intervals = pd.concat([self.test_intervals] * 10, ignore_index=True)
            
            # Test batch processing
            batch_size = 5
            for batch in reader.read(large_intervals, batch_size=batch_size):
                self.assertLessEqual(len(batch), batch_size)
                for seq in batch:
                    self.assertIsInstance(seq, str)
                    self.assertEqual(len(seq), 10)  # All test intervals are 10bp
                    
    def test_context_manager(self):
        """Test context manager behavior."""
        reader = FastaReader(self.fasta_file)
        
        # Test using with statement
        with reader as fasta:
            self.assertIsNotNone(fasta._fasta)
            sequence = fasta.get_sequence("chr1", 0, 10)
            self.assertEqual(sequence, "ACTGACTGAC")
            
        # File should be closed after context
        self.assertIsNone(reader._fasta)
        
        # Should still work with new context
        with reader as fasta:
            sequence = fasta.get_sequence("chr1", 0, 10)
            self.assertEqual(sequence, "ACTGACTGAC")

if __name__ == '__main__':
    unittest.main() 

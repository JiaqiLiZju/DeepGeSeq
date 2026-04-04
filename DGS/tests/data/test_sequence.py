"""Test module for sequence operations.

This module provides comprehensive tests for:
1. Core sequence manipulation functions
2. One-hot encoding/decoding operations
3. Sequence metrics (GC content, complexity)
4. Genome operations
5. Legacy function compatibility
6. Real-world usage scenarios
"""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from DGS.Data.Sequence import (
    # Core functions
    validate_sequence, get_reverse_complement,
    # One-hot encoding
    sequence_to_onehot, onehot_to_sequence,
    batch_to_onehot, batch_from_onehot,
    # Sequence metrics
    calculate_gc_content, calculate_sliding_gc,
    calculate_complexity,
    # Classes
    DNASeq, Genome,
    # Legacy functions
    reverse_complement, one_hot_encode, one_hot_decode
)

class TestCoreSequenceFunctions(unittest.TestCase):
    """Test core sequence manipulation functions."""
    
    def setUp(self):
        """Set up test sequences."""
        self.sequences = {
            'simple': 'ATCG',
            'with_n': 'ATCGN',
            'mixed_case': 'AtCgN',
            'empty': '',
            'invalid': 'ATCGX',
            'all_n': 'NNNN',
            'repetitive': 'AAAA',
            'long': 'ATCGATCGATCG'
        }
    
    def test_sequence_validation(self):
        """Test sequence validation function."""
        # Valid sequences
        for name, seq in self.sequences.items():
            if name != 'invalid':
                self.assertTrue(validate_sequence(seq))
        
        # Invalid sequences
        with self.assertRaises(ValueError):
            validate_sequence(self.sequences['invalid'])
        
        # Test more invalid cases
        invalid_seqs = ['AT CG', 'ATG-C', 'ATCG1', 'AT#CG']
        for seq in invalid_seqs:
            with self.assertRaises(ValueError):
                validate_sequence(seq)
    
    def test_reverse_complement(self):
        """Test reverse complement function."""
        test_cases = {
            'ATCG': 'CGAT',
            'NNNN': 'NNNN',
            'ATcgN': 'NcgAT',
            '': '',
            'AAAAAA': 'TTTTTT'
        }
        
        for input_seq, expected in test_cases.items():
            self.assertEqual(get_reverse_complement(input_seq), expected)
            # Test legacy function
            self.assertEqual(reverse_complement(input_seq), expected)
            
        # Test double reverse complement returns original
        seq = 'ATCGATCG'
        self.assertEqual(
            get_reverse_complement(get_reverse_complement(seq)),
            seq
        )

class TestOneHotEncoding(unittest.TestCase):
    """Test one-hot encoding and decoding operations."""
    
    def setUp(self):
        """Set up test sequences and expected encodings."""
        self.test_cases = {
            'A': np.array([[1,0,0,0]]),
            'C': np.array([[0,1,0,0]]),
            'G': np.array([[0,0,1,0]]),
            'T': np.array([[0,0,0,1]]),
            'N': np.array([[0,0,0,0]]),
            'ATCG': np.array([
                [1,0,0,0],
                [0,0,0,1],
                [0,1,0,0],
                [0,0,1,0]
            ])
        }
    
    def test_sequence_to_onehot(self):
        """Test single sequence one-hot encoding."""
        # Test each case
        for seq, expected in self.test_cases.items():
            encoded = sequence_to_onehot(seq)
            np.testing.assert_array_equal(encoded, expected)
        
        # Test empty sequence
        encoded = sequence_to_onehot('')
        self.assertEqual(encoded.shape, (0, 4))
        
        # Test different dtypes
        dtypes = [np.float32, np.float64, np.int32, np.int64]
        for dtype in dtypes:
            encoded = sequence_to_onehot('ATCG', dtype=dtype)
            self.assertEqual(encoded.dtype, dtype)
    
    def test_onehot_to_sequence(self):
        """Test one-hot decoding to sequence."""
        # Test each case
        for expected_seq, encoded in self.test_cases.items():
            decoded = onehot_to_sequence(encoded)
            self.assertEqual(decoded, expected_seq)
        
        # Test N handling
        n_encoded = np.zeros((3, 4))
        decoded = onehot_to_sequence(n_encoded)
        self.assertEqual(decoded, 'N')  # Multiple N's should collapse to single N
        
        # Test without N
        decoded = onehot_to_sequence(n_encoded, include_n=False)
        self.assertEqual(decoded, 'AAA')  # Should default to 'A'
    
    def test_batch_operations(self):
        """Test batch encoding and decoding."""
        # Test batch encoding
        sequences = ['ATCG', 'GCTA', 'NNNN', '']
        encoded = batch_to_onehot(sequences)
        self.assertEqual(encoded.shape, (4, 4, 4))
        
        # Test empty batch
        empty_batch = batch_to_onehot([])
        self.assertEqual(empty_batch.shape, (0, 0, 4))
        
        # Test batch with empty sequence
        single_empty = batch_to_onehot([''])
        self.assertEqual(single_empty.shape, (1, 0, 4))
        
        # Test mixed length sequences
        mixed_batch = batch_to_onehot(['A', 'AT', 'ATG'])
        self.assertEqual(mixed_batch.shape, (3, 3, 4))
        
        # Test roundtrip conversion
        sequences = ['ATCG', 'GCTA', 'NNNN', 'ATcgN']
        encoded = batch_to_onehot(sequences)
        decoded = batch_from_onehot(encoded)
        for orig, dec in zip(sequences, decoded):
            # Batch decoding keeps right-padding as trailing N.
            if set(orig.upper()) == {'N'}:
                self.assertEqual(dec, 'N')
            else:
                self.assertTrue(dec.startswith(orig.upper()))

class TestSequenceMetrics(unittest.TestCase):
    """Test sequence metric calculations."""
    
    def setUp(self):
        """Set up test sequences."""
        self.gc_test_cases = {
            'ATCG': 0.5,
            'GCGC': 1.0,
            'AAAA': 0.0,
            'ATCGN': 0.5,
            'NNNNN': 0.0,
            '': 0.0
        }
    
    def test_gc_content(self):
        """Test GC content calculation."""
        for seq, expected in self.gc_test_cases.items():
            self.assertAlmostEqual(calculate_gc_content(seq), expected)
    
    def test_sliding_gc(self):
        """Test sliding window GC content."""
        # Test with different window sizes
        seq = 'ATCGCGAT'
        
        # Window size 2
        gc2 = calculate_sliding_gc(seq, 2)
        expected2 = [0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0]
        np.testing.assert_array_almost_equal(gc2, expected2)
        
        # Window size 4
        gc4 = calculate_sliding_gc(seq, 4)
        expected4 = [0.5, 0.75, 1.0, 0.75, 0.5]
        np.testing.assert_array_almost_equal(gc4, expected4)
        
        # Test with N's
        seq_n = 'ATCGNNCG'
        gc_n = calculate_sliding_gc(seq_n, 3)
        expected_n = [1/3, 2/3, 1.0, 1.0, 1.0, 1.0]
        np.testing.assert_array_almost_equal(gc_n, expected_n)
    
    def test_sequence_complexity(self):
        """Test sequence complexity calculation."""
        # Test basic cases
        test_cases = {
            'AAAA': 1/3,  # Single unique 2-mer in 3 windows
            'ATCG': 1.0,   # All unique 2-mers
            'ATAT': 2/3,   # Two unique 2-mers in 3 windows
            'ATCGATCG': 4/7  # Four unique 2-mers in 7 windows
        }
        
        for seq, expected in test_cases.items():
            self.assertAlmostEqual(
                calculate_complexity(seq, k=2, normalize=True),
                expected,
                places=2
            )
        
        # Test different k values
        seq = 'ATCGATCGATCG'
        complexities = [
            calculate_complexity(seq, k=k, normalize=True)
            for k in range(1, 5)
        ]
        self.assertTrue(all(0.0 <= c <= 1.0 for c in complexities))
        
        # Test with N's
        seq_n = 'ATCGNNATCG'
        self.assertTrue(0 < calculate_complexity(seq_n, k=2) < 1)
        
        # Test edge cases
        self.assertEqual(calculate_complexity('', k=2), 0.0)
        self.assertEqual(calculate_complexity('AT', k=3), 0.0)
        self.assertEqual(calculate_complexity('NNNN', k=2), 0.0)

class TestDNASeqClass(unittest.TestCase):
    """Test DNASeq class functionality."""
    
    def setUp(self):
        """Set up test sequences."""
        self.sequences = {
            'simple': DNASeq('ATCG'),
            'with_n': DNASeq('ATCGN'),
            'empty': DNASeq(''),
            'repetitive': DNASeq('AAAA'),
            'long': DNASeq('ATCGATCGATCG')
        }
    
    def test_basic_operations(self):
        """Test basic sequence operations."""
        seq = self.sequences['simple']
        
        # Test string representation
        self.assertEqual(str(seq), 'ATCG')
        
        # Test length
        self.assertEqual(len(seq), 4)
        
        # Test reverse complement
        self.assertEqual(
            str(seq.reverse_complement()),
            'CGAT'
        )
    
    def test_sequence_metrics(self):
        """Test sequence metric calculations."""
        seq = self.sequences['simple']
        
        # Test GC content
        self.assertEqual(seq.gc_content(), 0.5)
        
        # Test sliding window GC
        gc_windows = seq.gc_content(window_size=2)
        expected = [0.0, 0.5, 1.0]
        np.testing.assert_array_almost_equal(gc_windows, expected)
        
        # Test complexity
        self.assertTrue(0 < seq.complexity(k=2) <= 1)
    
    def test_onehot_conversion(self):
        """Test one-hot conversion methods."""
        seq = self.sequences['simple']
        
        # Test to_onehot
        encoded = seq.to_onehot()
        self.assertEqual(encoded.shape, (4, 4))
        
        # Test from_onehot
        decoded = DNASeq.from_onehot(encoded)
        self.assertEqual(str(decoded), str(seq))

class TestGenomeClass(unittest.TestCase):
    """Test Genome class functionality."""
    
    def setUp(self):
        """Set up test genome."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test FASTA
        self.fasta_path = os.path.join(self.temp_dir, 'test.fa')
        with open(self.fasta_path, 'w') as f:
            f.write('>chr1\nATCGATCGATCGN\n')
            f.write('>chr2\nTGCANNNNTGCA\n')
        
        # Create test intervals
        self.intervals = pd.DataFrame({
            'chrom': ['chr1', 'chr1', 'chr2'],
            'start': [0, 5, 0],
            'end': [4, 9, 4],
            'strand': ['+', '-', '+']
        })
    
    def tearDown(self):
        """Clean up test files."""
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    def test_context_manager(self):
        """Test reader lifecycle and explicit close behavior."""
        genome = Genome(self.fasta_path)
        self.assertIsNotNone(genome._reader)
        genome.close()
        self.assertIsNone(genome._reader)
    
    def test_sequence_extraction(self):
        """Test sequence extraction."""
        genome = Genome(self.fasta_path)
        try:
            # Test basic extraction
            seqs = genome.extract_sequences(self.intervals)
            self.assertEqual(len(seqs), 3)

            # Test strand-aware extraction
            seqs_stranded = genome.extract_sequences(
                self.intervals,
                strand_aware=True
            )
            # Second sequence should be reverse complemented
            self.assertEqual(
                str(seqs_stranded[1]),
                str(seqs[1].reverse_complement())
            )
        finally:
            genome.close()
    
    def test_invalid_extraction(self):
        """Test invalid sequence extraction."""
        genome = Genome(self.fasta_path)
        try:
            # Test invalid chromosome
            invalid_intervals = pd.DataFrame({
                'chrom': ['chr3'],
                'start': [0],
                'end': [4]
            })
            with self.assertRaises(KeyError):
                genome.extract_sequences(invalid_intervals)

            # Test invalid coordinates
            invalid_coords = pd.DataFrame({
                'chrom': ['chr1'],
                'start': [-1],
                'end': [5]
            })
            with self.assertRaises(ValueError):
                genome.extract_sequences(invalid_coords)
        finally:
            genome.close()

def usage_example():
    """Demonstrate typical usage of sequence operations."""
    print("\n=== Sequence Operations Usage Example ===\n")
    
    # 1. Basic sequence manipulation
    print("1. Basic Sequence Manipulation:")
    seq = DNASeq('ATCGATCG')
    print(f"Original sequence: {seq}")
    print(f"Reverse complement: {seq.reverse_complement()}")
    print(f"GC content: {seq.gc_content():.2f}")
    print(f"Sequence complexity (k=2): {seq.complexity(k=2):.2f}")
    
    # 2. One-hot encoding
    print("\n2. One-hot Encoding:")
    encoded = seq.to_onehot()
    print(f"Shape of one-hot encoded sequence: {encoded.shape}")
    decoded = DNASeq.from_onehot(encoded)
    print(f"Decoded sequence: {decoded}")
    
    # 3. Batch operations
    print("\n3. Batch Operations:")
    sequences = ['ATCG', 'GCTA', 'NNNN']
    print(f"Original sequences: {sequences}")
    batch_encoded = batch_to_onehot(sequences)
    print(f"Shape of batch encoding: {batch_encoded.shape}")
    batch_decoded = batch_from_onehot(batch_encoded)
    print(f"Decoded sequences: {batch_decoded}")
    
    # 4. Genome operations
    print("\n4. Genome Operations:")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test FASTA
        fasta_path = os.path.join(temp_dir, 'example.fa')
        with open(fasta_path, 'w') as f:
            f.write('>chr1\nATCGATCGATCGN\n')
            f.write('>chr2\nTGCANNNNTGCA\n')
        
        # Create intervals
        intervals = pd.DataFrame({
            'chrom': ['chr1', 'chr2'],
            'start': [0, 4],
            'end': [4, 8],
            'strand': ['+', '-']
        })
        
        # Extract and process sequences
        with Genome(fasta_path) as genome:
            sequences = genome.extract_sequences(intervals)
            
            print("Extracted sequences:")
            for i, seq in enumerate(sequences):
                print(f"\nSequence {i+1}:")
                print(f"Sequence: {seq}")
                print(f"Length: {len(seq)}")
                print(f"GC content: {seq.gc_content():.2f}")
                print(f"Sliding window GC content (window=3):")
                print(seq.gc_content(window_size=3))

if __name__ == '__main__':
    # Run the usage example
    print("Running usage example...")
    usage_example()
    
    # Run the tests
    unittest.main(argv=[''], verbosity=2, exit=False) 

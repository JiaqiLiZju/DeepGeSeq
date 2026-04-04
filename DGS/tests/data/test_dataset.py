"""Tests for dataset classes and utilities.

This module provides comprehensive unit tests for:
1. SeqDataset
2. GenomicDataset
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
try:
    import pyBigWig
except ImportError:  # pragma: no cover - optional dependency
    pyBigWig = None

from DGS.Data.Dataset import SeqDataset, GenomicDataset
from DGS.Data.Interval import Interval
from DGS.Data.Sequence import DNASeq
from DGS.Data.Target import Target

class TestSeqDataset(unittest.TestCase):
    """Test sequence dataset functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock genome with known sequences
        self.test_sequences = {
            'chr1': 'ATCGATCGATCGATCG',
            'chr2': 'GCTAGCTAGCTAGCTA'
        }
        self.genome = type('MockGenome', (), {
            'extract_sequences': lambda self, intervals, strand_aware: [
                DNASeq(self.test_sequences[row['chrom']][row['start']:row['end']])
                for _, row in intervals.iterrows()
            ]
        })()
        self.genome.test_sequences = self.test_sequences
        
        # Create test intervals
        self.intervals = Interval(pd.DataFrame({
            'chrom': ['chr1', 'chr1', 'chr2'],
            'start': [0, 4, 0],
            'end': [8, 12, 8],
            'strand': ['+', '-', '+']
        }))
    
    def test_init(self):
        """Test initialization."""
        dataset = SeqDataset(self.intervals, self.genome)
        self.assertEqual(len(dataset), 3)
        
        # Test with strand_aware=False
        dataset_no_strand = SeqDataset(self.intervals, self.genome, strand_aware=False)
        self.assertEqual(len(dataset_no_strand), 3)
    
    def test_getitem(self):
        """Test item retrieval."""
        dataset = SeqDataset(self.intervals, self.genome)
        
        # Test first sequence (forward strand)
        seq0 = dataset[0]
        self.assertIsInstance(seq0, np.ndarray)
        self.assertEqual(seq0.shape, (8, 4))  # 8 bases, 4 nucleotides
        
        # Test second sequence (reverse strand)
        seq1 = dataset[1]
        self.assertEqual(seq1.shape, (8, 4))
        
        # Test third sequence (different chromosome)
        seq2 = dataset[2]
        self.assertEqual(seq2.shape, (8, 4))
        
    def test_sequence_extraction(self):
        """Test sequence extraction functionality."""
        dataset = SeqDataset(self.intervals, self.genome)
        seqs = dataset.seqs
        
        # Test number of sequences
        self.assertEqual(len(seqs), 3)
        
        # Test first sequence content
        self.assertEqual(str(seqs[0]), self.test_sequences['chr1'][:8])
        
        # Test second sequence content
        self.assertEqual(str(seqs[1]), self.test_sequences['chr1'][4:12])
        
        # Test third sequence content
        self.assertEqual(str(seqs[2]), self.test_sequences['chr2'][:8])
        
    def test_invalid_intervals(self):
        """Test handling of invalid intervals."""
        # Test with invalid chromosome
        invalid_chrom = Interval(pd.DataFrame({
            'chrom': ['chr3'],  # Invalid chromosome
            'start': [0],
            'end': [8],
            'strand': ['+']
        }))
        with self.assertRaises(Exception):
            SeqDataset(invalid_chrom, self.genome)
        
        # Test with invalid coordinates - we'll create the interval directly
        # to bypass the Interval validation
        invalid_coords = pd.DataFrame({
            'chrom': ['chr1'],
            'start': [-1],  # Invalid start position
            'end': [8],
            'strand': ['+']
        })
        with self.assertRaises(ValueError):
            Interval(invalid_coords)
        
        # Test with end <= start
        invalid_range = pd.DataFrame({
            'chrom': ['chr1'],
            'start': [8],
            'end': [8],  # end = start
            'strand': ['+']
        })
        with self.assertRaises(ValueError):
            Interval(invalid_range)
            
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        empty_intervals = Interval(pd.DataFrame({
            'chrom': [],
            'start': [],
            'end': [],
            'strand': []
        }))
        dataset = SeqDataset(empty_intervals, self.genome)
        self.assertEqual(len(dataset), 0)

@unittest.skipUnless(pyBigWig is not None, "pyBigWig is required for GenomicDataset tests")
class TestGenomicDataset(unittest.TestCase):
    """Test genomic dataset functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock genome with known sequences
        self.test_sequences = {
            'chr1': 'ATCGATCGATCGATCG',
            'chr2': 'GCTAGCTAGCTAGCTA'
        }
        self.genome = type('MockGenome', (), {
            'extract_sequences': lambda self, intervals, strand_aware: [
                DNASeq(self.test_sequences[row['chrom']][row['start']:row['end']])
                for _, row in intervals.iterrows()
            ]
        })()
        self.genome.test_sequences = self.test_sequences
        
        # Create test intervals
        self.intervals = Interval(pd.DataFrame({
            'chrom': ['chr1', 'chr1', 'chr2'],
            'start': [0, 4, 0],
            'end': [8, 12, 8],
            'strand': ['+', '-', '+']
        }))
        
        # Create temporary files for targets
        self.temp_dir = tempfile.mkdtemp()
        self.bed_file = os.path.join(self.temp_dir, 'test.bed')
        self.bigwig_file = os.path.join(self.temp_dir, 'test.bw')
        
        # Create mock target data
        self.target_data = pd.DataFrame({
            'chrom': ['chr1', 'chr2'],
            'start': [0, 0],
            'end': [12, 8],
            'value': [1, 0]
        })
        self.target_data.to_csv(self.bed_file, sep='\t', index=False, header=False)

        # Create a mock BigWig file
        with pyBigWig.open(self.bigwig_file, 'w') as bw:
            bw.addHeader([("chr1", 20), ("chr2", 20)])  # Add chromosomes
            bw.addEntries(["chr1"]*3, [0, 5, 10], ends=[1, 6, 11], values=[1.0, 2.0, 3.0])
            bw.addEntries(["chr2"]*3, [0, 5, 10], ends=[1, 6, 11], values=[5.0, 6.0, 7.0])
        
        
        # Create mock targets
        self.targets = Target(self.intervals.data, [
            {
                'task_name': 'binary_task',
                'file_path': self.bed_file,
                'file_type': 'bed',
                'task_type': 'binary',
                'target_column': 'value'
            },
            {
                'task_name': 'regression_task',
                'file_path': self.bigwig_file,
                'file_type': 'bigwig',
                'task_type': 'regression',
                'bin_size': None
            }
        ])
        # Mock the target data
        self.targets.data = {
            'binary_task': np.array([1, 1, 0], dtype=np.int8),
            'regression_task': np.array([1, 1, 0], dtype=np.float32)
        }
        
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization."""
        dataset = GenomicDataset(self.intervals, self.genome, self.targets)
        self.assertEqual(len(dataset), 3)
        self.assertIsNotNone(dataset.task_info)
        self.assertEqual(len(dataset.task_info), 2)  # Two tasks
        
    def test_getitem(self):
        """Test item retrieval."""
        dataset = GenomicDataset(self.intervals, self.genome, self.targets)
        
        # Test first item
        seq, labels = dataset[0]
        self.assertIsInstance(seq, np.ndarray)
        self.assertEqual(seq.shape, (8, 4))  # 8 bases, 4 nucleotides
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(labels.shape, (2,))  # 2 tasks
        self.assertEqual(labels.dtype, np.float32)
        
        # Test all items
        for i in range(len(dataset)):
            seq, labels = dataset[i]
            self.assertEqual(seq.shape, (8, 4))
            self.assertEqual(labels.shape, (2,))
            
    def test_sequence_extraction(self):
        """Test sequence extraction functionality."""
        dataset = GenomicDataset(self.intervals, self.genome, self.targets)
        seqs = dataset.seqs
        
        # Test number of sequences
        self.assertEqual(len(seqs), 3)
        
        # Test sequence contents
        self.assertEqual(str(seqs[0]), self.test_sequences['chr1'][:8])
        self.assertEqual(str(seqs[1]), self.test_sequences['chr1'][4:12])
        self.assertEqual(str(seqs[2]), self.test_sequences['chr2'][:8])
        
    def test_label_extraction(self):
        """Test label extraction functionality."""
        dataset = GenomicDataset(self.intervals, self.genome, self.targets)
        
        # Test binary task labels
        binary_labels = dataset.targets.get_labels('binary_task')
        self.assertEqual(binary_labels.shape, (3,))
        self.assertEqual(binary_labels.dtype, np.int8)
        np.testing.assert_array_equal(binary_labels, [1, 1, 0])
        
        # Test regression task labels
        regression_labels = dataset.targets.get_labels('regression_task')
        self.assertEqual(regression_labels.shape, (3,))
        self.assertEqual(regression_labels.dtype, np.float32)

    def test_bigwig_label_extraction(self):
        """Test extraction of labels from BigWig file."""
        # Create a new target for the BigWig file
        bigwig_target = Target(self.intervals.data, [
            {
                'task_name': 'bigwig_task',
                'file_path': self.bigwig_file,
                'file_type': 'bigwig',
                'task_type': 'regression',
                'bin_size': None
            }
        ])
        
        # Create a new dataset with the BigWig target
        dataset = GenomicDataset(self.intervals, self.genome, bigwig_target)
        
        # Test extraction of regression task labels
        regression_labels = dataset.targets.get_labels('bigwig_task')
        print(regression_labels)
        self.assertEqual(regression_labels.shape, (3, 1))  # 3 intervals, 1 value each
        self.assertTrue(np.all(regression_labels >= 0.1))  # Check that values are as expected

    def test_task_info(self):
        """Test task information retrieval."""
        dataset = GenomicDataset(self.intervals, self.genome, self.targets)
        task_info = dataset.task_info
        print(task_info.T)
        
        # Check task names
        self.assertIn('binary_task', task_info.task_name)
        self.assertIn('regression_task', task_info.task_name)
        
        # Check task types
        self.assertEqual(task_info.loc['binary_task']['task_type'], 'binary')
        self.assertEqual(task_info.loc['regression_task']['task_type'], 'regression')
        
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        empty_intervals = Interval(pd.DataFrame({
            'chrom': [],
            'start': [],
            'end': [],
            'strand': []
        }))
        
        # Create empty targets with file paths
        empty_targets = Target(empty_intervals.data, [
            {
                'task_name': 'binary_task',
                'file_path': self.bed_file,
                'file_type': 'bed',
                'task_type': 'binary'
            }
        ])
        
        dataset = GenomicDataset(empty_intervals, self.genome, empty_targets)
        self.assertEqual(len(dataset), 0)
        
    def test_invalid_targets(self):
        """Test handling of invalid targets."""
        # Test with mismatched number of labels
        invalid_targets = self.targets
        invalid_targets.data['binary_task'] = np.array([1, 0])  # Wrong length
        
        with self.assertRaises(Exception):
            GenomicDataset(self.intervals, self.genome, invalid_targets)

if __name__ == '__main__':
    unittest.main()

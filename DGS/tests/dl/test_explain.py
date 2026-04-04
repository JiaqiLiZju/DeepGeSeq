"""Unit tests for test explain."""

import unittest
import numpy as np
import torch
import tempfile
from pathlib import Path
import os
import shutil
import importlib.util

from DGS.DL.Explain import motif_enrich, Seqlet_Calling

HAS_TANGERMEME = importlib.util.find_spec("tangermeme") is not None
HAS_MODISCO = shutil.which("modisco") is not None

class SimpleModel(torch.nn.Module):
    """SimpleModel implementation."""
    def __init__(self):
        """Initialize `SimpleModel`."""
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 16, kernel_size=8)
        self.fc = torch.nn.Linear(16, 1)
        
    def forward(self, x):
        """Compute forward outputs for `SimpleModel`."""
        x = self.conv(x)
        x = torch.mean(x, dim=2)
        return self.fc(x)

class TestExplain(unittest.TestCase):
    """Test cases for explain."""
    def setUp(self):
        """Set up test data and model"""
        # Set random seeds
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create test data
        self.batch_size = 10
        self.seq_length = 500
        # X must be one-hot encoded
        self.idx = torch.randint(0, 4, (self.batch_size, self.seq_length))
        self.X = torch.zeros((self.batch_size, self.seq_length, 4))
        for i in range(self.batch_size):
            for j in range(self.seq_length):
                self.X[i, j, self.idx[i, j]] = 1
        
        # Create model
        self.model = SimpleModel()
        
        # Create temporary directory
        self.temp_dir = "." #tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        
        # Create a simple MEME format motif database for testing
        self.motif_db = self.output_dir / "test_motifs.meme"
        with open(self.motif_db, "w") as f:
            f.write("""MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

MOTIF M1 Test_Motif
letter-probability matrix: alength= 4 w= 8
0.8  0.1  0.05 0.05
0.05 0.8  0.1  0.05
0.1  0.05 0.8  0.05
0.05 0.1  0.05 0.8
0.8  0.05 0.1  0.05
0.05 0.8  0.05 0.1
0.1  0.05 0.8  0.05
0.05 0.1  0.05 0.8
""")

    @unittest.skipUnless(
        HAS_TANGERMEME and HAS_MODISCO,
        "motif_enrich requires tangermeme and modisco",
    )
    def test_motif_enrich(self):
        """Test motif enrichment analysis"""
        motifs_file = motif_enrich(
            self.model,
            self.X,
            target=0,
            output_dir=str(self.output_dir / "motif_results"),
            max_seqlets=100
        )
        
        # Check if output files exist
        self.assertTrue(os.path.exists(self.output_dir / "motif_results" / "ohe.npz"))
        self.assertTrue(os.path.exists(self.output_dir / "motif_results" / "shap.npz"))
        self.assertTrue(os.path.exists(motifs_file))
        
        # Load and check results
        ohe_data = np.load(self.output_dir / "motif_results" / "ohe.npz")
        shap_data = np.load(self.output_dir / "motif_results" / "shap.npz")
        
        self.assertEqual(ohe_data['arr_0'].shape[0], self.batch_size)
        self.assertEqual(shap_data['arr_0'].shape[0], self.batch_size)
            

    @unittest.skipUnless(HAS_TANGERMEME, "Seqlet_Calling requires tangermeme")
    def test_seqlet_calling(self):
        """Test seqlet calling and annotation"""
        seqlet_info = Seqlet_Calling(
            self.model,
            self.X,
            target=0,
            output_dir=str(self.output_dir / "seqlet_results"),
            motif_db=str(self.motif_db)
        )
        
        # Check if output files exist
        self.assertTrue(os.path.exists(self.output_dir / "seqlet_results" / "seqlets.npz"))
        self.assertTrue(os.path.exists(self.output_dir / "seqlet_results" / "annotations.npz"))
        
        # Check seqlet_info contents
        required_keys = ['example_idx', 'start', 'end', 'motif_indices', 'motif_pvalues']
        for key in required_keys:
            self.assertIn(key, seqlet_info)
            self.assertIsInstance(seqlet_info[key], np.ndarray)
        
        # Check array shapes
        n_seqlets = len(seqlet_info['example_idx'])
        self.assertEqual(len(seqlet_info['start']), n_seqlets)
        self.assertEqual(len(seqlet_info['end']), n_seqlets)
        self.assertEqual(len(seqlet_info['motif_indices']), n_seqlets)
        self.assertEqual(len(seqlet_info['motif_pvalues']), n_seqlets)
        
        # Check value ranges
        self.assertTrue(all(0 <= idx < self.batch_size for idx in seqlet_info['example_idx']))
        self.assertTrue(all(0 <= start < self.seq_length for start in seqlet_info['start']))
        self.assertTrue(all(0 < end <= self.seq_length for end in seqlet_info['end']))
        self.assertTrue(all(start < end for start, end in zip(seqlet_info['start'], seqlet_info['end'])))
            

    @unittest.skipUnless(HAS_TANGERMEME, "Explain error handling requires tangermeme")
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with invalid model
        with self.assertRaises(Exception):
            motif_enrich(None, self.X, target=0)
            
        # Test with invalid input tensor
        with self.assertRaises(Exception):
            motif_enrich(self.model, None, target=0)
            
        # Test with invalid motif database
        with self.assertRaises(Exception):
            Seqlet_Calling(self.model, self.X, target=0, motif_db="nonexistent.meme")

    def tearDown(self):
        """Clean up temporary files"""
        #shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main() 

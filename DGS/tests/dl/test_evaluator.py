"""Unit tests for test evaluator."""

import unittest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from DGS.DL.Evaluator import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    calculate_sequence_classification_metrics,
    calculate_sequence_regression_metrics,
    show_auc_curve,
    show_pr_curve
)

class TestEvaluator(unittest.TestCase):
    """Test cases for evaluator."""
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Binary classification data
        self.n_samples = 100
        self.y_true_binary = np.random.randint(0, 2, (self.n_samples,))
        self.y_pred_binary = np.random.random((self.n_samples,))
        
        # Multiclass data
        self.n_classes = 3
        self.y_true_multi = np.random.randint(0, self.n_classes, (self.n_samples,))
        self.y_pred_multi = np.random.random((self.n_samples, self.n_classes))
        self.y_pred_multi = self.y_pred_multi / self.y_pred_multi.sum(axis=1, keepdims=True)
        
        # Sequence data
        self.seq_len = 10
        self.n_seq_classes = 4
        self.y_true_seq = np.random.randint(0, 1, (self.n_samples, self.seq_len, self.n_seq_classes))
        self.y_pred_seq = np.random.random((self.n_samples, self.seq_len, self.n_seq_classes))
        self.y_pred_seq = self.y_pred_seq / self.y_pred_seq.sum(axis=2, keepdims=True)
        
        # Regression data
        self.y_true_reg = np.random.randn(self.n_samples)
        self.y_pred_reg = self.y_true_reg + np.random.randn(self.n_samples) * 0.1

        # multi-Regression data
        self.y_true_multi_reg = np.random.randn(self.n_samples, self.n_classes)
        self.y_pred_multi_reg = self.y_true_multi_reg + np.random.randn(self.n_samples, self.n_classes) * 0.1

        # sequence-regression data
        self.y_true_seq_reg = np.random.randn(self.n_samples, self.seq_len, self.n_seq_classes)
        self.y_pred_seq_reg = self.y_true_seq_reg + np.random.randn(self.n_samples, self.seq_len, self.n_seq_classes) * 0.1

        # Create temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

    def test_binary_classification_metrics(self):
        """Test binary classification metrics"""
        # Test default DataFrame output
        df = calculate_classification_metrics(self.y_true_binary, self.y_pred_binary)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        print(df)

        # Check required metrics exist
        required_metrics = ['accuracy', 'auroc', 'auprc', 'f1']
        for metric in required_metrics:
            self.assertIn(metric, df.columns)
            self.assertTrue(0 <= df[metric].iloc[0] <= 1)
        
        # Test dictionary output
        metrics = calculate_classification_metrics(
            self.y_true_binary, 
            self.y_pred_binary,
            return_dict=True
        )
        self.assertIsInstance(metrics, dict)
        for metric in required_metrics:
            self.assertIn(metric, metrics)

    def test_multiclass_classification_metrics(self):
        """Test multiclass classification metrics"""
        # Test default DataFrame output
        df = calculate_classification_metrics(self.y_true_multi, self.y_pred_multi)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        print(df)
        
        # Check per-class metrics exist
        for i in range(self.n_classes):
            class_prefix = f'task_{i}'
            self.assertTrue(any(col.startswith(class_prefix) for col in df.index))
            
        # Test dictionary output
        metrics = calculate_classification_metrics(
            self.y_true_multi, 
            self.y_pred_multi,
            return_dict=True
        )
        self.assertIsInstance(metrics, dict)
        for i in range(self.n_classes):
            self.assertIn(f'task_{i}', metrics)

    def test_regression_metrics(self):
        """Test regression metrics"""
        # Test default DataFrame output
        df = calculate_regression_metrics(self.y_true_reg, self.y_pred_reg)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        print(df)

        # Check required metrics exist
        required_metrics = ['mse', 'rmse', 'mae', 'r2', 'pearson_r']
        for metric in required_metrics:
            self.assertIn(metric, df.columns)
        
        # Test dictionary output
        metrics = calculate_regression_metrics(
            self.y_true_reg, 
            self.y_pred_reg,
            return_dict=True
        )
        self.assertIsInstance(metrics, dict)
        for metric in required_metrics:
            self.assertIn(metric, metrics)

    def test_multi_regression_metrics(self):
        """Test multi-regression metrics"""
        # Test default DataFrame output
        df = calculate_regression_metrics(self.y_true_multi_reg, self.y_pred_multi_reg)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), self.n_classes)
        print(df)

        # Check required metrics exist
        required_metrics = ['mse', 'rmse', 'mae', 'r2', 'pearson_r']
        for metric in required_metrics:
            self.assertIn(metric, df.columns)
        
        # Test dictionary output
        metrics = calculate_regression_metrics(
            self.y_true_multi_reg, 
            self.y_pred_multi_reg,
            return_dict=True
        )
        self.assertIsInstance(metrics, dict)
        for i in range(self.n_classes):
            self.assertIn(f'task_{i}', metrics)

    def test_sequence_classification_metrics(self):
        """Test sequence classification metrics"""
        # Test default DataFrame output
        df, df_per_sample = calculate_sequence_classification_metrics(
            self.y_true_seq,
            self.y_pred_seq
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df_per_sample, pd.DataFrame)
        print(df_per_sample)
        print(df)

        # Check overall metrics
        self.assertIn('accuracy', df.columns)
        self.assertIn('auroc', df.columns)
        
        # Check per-sample metrics
        self.assertTrue(len(df_per_sample.columns) > 0)
        self.assertTrue(any('sample_0' in col for col in df_per_sample.index))
        
        # Test with mask
        mask = np.random.random((self.n_samples, self.seq_len)) > 0.2
        df_masked, _ = calculate_sequence_classification_metrics(
            self.y_true_seq,
            self.y_pred_seq,
            mask=mask
        )
        self.assertIsInstance(df_masked, pd.DataFrame)

    def test_sequence_regression_metrics(self):
        """Test sequence regression metrics"""
        # Create sequence regression data
        y_true_seq_reg = self.y_true_seq_reg
        y_pred_seq_reg = self.y_pred_seq_reg
        
        # Test default DataFrame output
        df, df_per_sample = calculate_sequence_regression_metrics(
            y_true_seq_reg,
            y_pred_seq_reg
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df_per_sample, pd.DataFrame)
        print(df_per_sample)
        print(df)

        # Check overall metrics
        required_metrics = ['mse', 'rmse', 'mae', 'r2']
        for metric in required_metrics:
            self.assertIn(metric, df.columns)
        
        # Check per-sample metrics
        self.assertTrue(len(df_per_sample.columns) > 0)
        self.assertTrue(any('sample_0' in col for col in df_per_sample.index))
        
        # Test with mask
        mask = np.random.random((self.n_samples, self.seq_len)) > 0.2
        df_masked, _ = calculate_sequence_regression_metrics(
            y_true_seq_reg,
            y_pred_seq_reg,
            mask=mask
        )
        self.assertIsInstance(df_masked, pd.DataFrame)

    def test_visualization_functions(self):
        """Test visualization functions"""
        # Get metrics for visualization
        metrics = calculate_classification_metrics(
            self.y_true_multi, 
            self.y_pred_multi,
            return_dict=True
        )
        
        # Test ROC curve
        show_auc_curve(
            metrics,
            save=True,
            output_dir=self.output_dir,
            output_fname='test_roc.pdf'
        )
        self.assertTrue((self.output_dir / 'test_roc.pdf').exists())
        
        # Test PR curve
        show_pr_curve(
            metrics,
            save=True,
            output_dir=self.output_dir,
            output_fname='test_pr.pdf'
        )
        self.assertTrue((self.output_dir / 'test_pr.pdf').exists())

    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main() 

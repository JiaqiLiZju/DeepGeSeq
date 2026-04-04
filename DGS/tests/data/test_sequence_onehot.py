"""Unit tests for one-hot sequence encoding helpers."""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from DGS.Data.Sequence import sequence_to_onehot, batch_to_onehot


def _legacy_onehot(seq: str) -> np.ndarray:
    """Internal helper for legacy onehot."""
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0],
    }
    if not seq:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array([mapping.get(base.upper(), [0, 0, 0, 0]) for base in seq], dtype=np.float32)


class TestSequenceOneHot(unittest.TestCase):
    """Test cases for sequence one hot."""
    def test_sequence_to_onehot_matches_legacy_behavior(self):
        """Test sequence to onehot matches legacy behavior."""
        sequences = [
            "",
            "ACGTN",
            "aaaacccgttn",
            "AXTG",
            "NNNN",
        ]
        for seq in sequences:
            with self.subTest(seq=seq):
                expected = _legacy_onehot(seq)
                actual = sequence_to_onehot(seq)
                self.assertEqual(actual.dtype, np.float32)
                self.assertEqual(actual.shape, expected.shape)
                np.testing.assert_array_equal(actual, expected)

    def test_batch_to_onehot_padding_and_values(self):
        """Test batch to onehot padding and values."""
        sequences = ["ACGT", "AT", ""]
        expected = np.zeros((3, 4, 4), dtype=np.float32)
        expected[0, :4] = _legacy_onehot("ACGT")
        expected[1, :2] = _legacy_onehot("AT")

        actual = batch_to_onehot(sequences)
        self.assertEqual(actual.shape, (3, 4, 4))
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()

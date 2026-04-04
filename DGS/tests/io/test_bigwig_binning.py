"""Unit tests for BigWig binning behavior."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock
import types
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from DGS.IO.bigwig import BigWigReader


class _FakeBigWigHandle:
    """Helper class used by local tests."""
    def __enter__(self):
        """Enter the runtime context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context."""
        return False

    def values(self, chrom, start, end, numpy=True):
        """Return interval values for the fake BigWig handle."""
        return np.array([1, 2, 3, 4, 5], dtype=np.float32)


class TestBigWigBinning(unittest.TestCase):
    """Test cases for big wig binning."""
    def test_bin_size_handles_tail_without_reshape_failure(self):
        """Test bin size handles tail without reshape failure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bw_path = Path(tmp_dir) / "test.bw"
            bw_path.write_bytes(b"dummy")
            reader = BigWigReader(bw_path)
            intervals = pd.DataFrame([{"chrom": "chr1", "start": 0, "end": 5}])

            fake_pybigwig = types.SimpleNamespace(open=lambda *_args, **_kwargs: _FakeBigWigHandle())
            with mock.patch.dict("sys.modules", {"pyBigWig": fake_pybigwig}):
                result = reader.read(intervals, bin_size=2, aggfunc="mean")

        self.assertEqual(result.shape, (1, 3))
        np.testing.assert_allclose(result[0], np.array([1.5, 3.5, 5.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

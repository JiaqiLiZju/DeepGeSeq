"""Unit tests for BED label alignment with non-consecutive interval indices."""

import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from DGS.Data.Target import Target


class TestTargetBedAlignment(unittest.TestCase):
    """Ensure BED overlaps map labels to the correct base intervals."""

    def test_non_consecutive_index_maps_labels_correctly(self):
        """Label assignment should use query_index, not overlap row order."""
        intervals = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [100, 300],
                "end": [200, 400],
            },
            index=[10, 11],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            bed_path = Path(tmp_dir) / "target.bed"
            bed_path.write_text("chr1\t320\t330\t1\n", encoding="utf-8")
            tasks = [
                {
                    "task_name": "peak",
                    "file_path": str(bed_path),
                    "file_type": "bed",
                    "target_column": "name",
                }
            ]
            target = Target(intervals, tasks)

        labels = target.get_labels("peak")
        np.testing.assert_array_equal(labels, np.array([0, 1], dtype=np.int8))


if __name__ == "__main__":
    unittest.main()


"""Unit tests for overlap index contract and merge behavior."""

import unittest
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from DGS.Data.Interval import Interval, find_overlaps


class TestIntervalOverlapContract(unittest.TestCase):
    """Test overlap output fields and merge semantics."""

    def test_find_overlaps_includes_stable_indices(self):
        """`find_overlaps` should include original query/target indices."""
        query = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [100, 300],
                "end": [200, 400],
                "score": [7, 9],
            },
            index=[10, 20],
        )
        target = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [150, 320],
                "end": [180, 330],
                "name": [1, 2],
            },
            index=[101, 305],
        )

        overlaps = find_overlaps(query, target)
        self.assertIn("query_index", overlaps.columns)
        self.assertIn("target_index", overlaps.columns)
        self.assertIn("query_score", overlaps.columns)
        self.assertEqual(set(overlaps["query_index"].tolist()), {10, 20})
        self.assertEqual(set(overlaps["target_index"].tolist()), {101, 305})

    def test_merge_with_left_right_outer_are_index_stable(self):
        """`merge_with` should use stable index fields for all merge modes."""
        left = Interval(
            pd.DataFrame(
                {
                    "chrom": ["chr1", "chr1"],
                    "start": [0, 20],
                    "end": [10, 30],
                },
                index=[10, 11],
            )
        )
        right = Interval(
            pd.DataFrame(
                {
                    "chrom": ["chr1", "chr1"],
                    "start": [5, 40],
                    "end": [8, 50],
                },
                index=[99, 100],
            )
        )

        left_merge = left.merge_with(right, how="left")
        right_merge = left.merge_with(right, how="right")
        outer_merge = left.merge_with(right, how="outer")

        self.assertEqual(len(left_merge), 2)
        self.assertEqual(len(right_merge), 2)
        self.assertEqual(len(outer_merge), 3)
        self.assertIn(99, set(left_merge["target_index"].dropna().tolist()))


if __name__ == "__main__":
    unittest.main()


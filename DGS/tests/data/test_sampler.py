"""Tests for interval splitting and dataloader helpers in DGS."""

import numpy as np
import pandas as pd
import pytest

from DGS.Data.Interval import Interval
from DGS.Data.Sampler import random_split, chromosome_split
from DGS.Data.Dataset import SeqDataset, create_dataloader
from DGS.Data.Sequence import DNASeq


class MockGenome:
    """Minimal genome stub for SeqDataset tests."""

    def extract_sequences(self, intervals, strand_aware=True):
        del strand_aware
        seqs = []
        for _, row in intervals.iterrows():
            length = int(row["end"] - row["start"])
            seq = ("ATCG" * ((length + 3) // 4))[:length]
            seqs.append(DNASeq(seq))
        return seqs


@pytest.fixture
def intervals():
    data = pd.DataFrame(
        {
            "chrom": ["chr1"] * 4 + ["chr2"] * 3 + ["chr3"] * 3,
            "start": np.arange(0, 120, 12),
            "end": np.arange(12, 132, 12),
        }
    )
    return Interval(data)


def _index_set(x):
    return set(x.data.index.tolist())


def test_random_split_three_way_reproducible(intervals):
    train, val, test = random_split(
        intervals,
        test_size=0.3,
        val_size=0.2,
        random_state=42,
    )
    train2, val2, test2 = random_split(
        intervals,
        test_size=0.3,
        val_size=0.2,
        random_state=42,
    )

    assert len(test.data) == 3
    assert len(val.data) == 2
    assert len(train.data) == 5

    assert train.data.index.tolist() == train2.data.index.tolist()
    assert val.data.index.tolist() == val2.data.index.tolist()
    assert test.data.index.tolist() == test2.data.index.tolist()

    idx_train = _index_set(train)
    idx_val = _index_set(val)
    idx_test = _index_set(test)
    idx_all = _index_set(intervals)

    assert idx_train.isdisjoint(idx_val)
    assert idx_train.isdisjoint(idx_test)
    assert idx_val.isdisjoint(idx_test)
    assert idx_train | idx_val | idx_test == idx_all


def test_random_split_two_way_contract(intervals):
    train, test = random_split(intervals, test_size=0.25, random_state=7)

    assert len(train.data) + len(test.data) == len(intervals.data)
    assert _index_set(train).isdisjoint(_index_set(test))


def test_chromosome_split_two_way(intervals):
    train, test = chromosome_split(intervals, test_chroms=["chr3"])

    assert all(train.data["chrom"] != "chr3")
    assert all(test.data["chrom"] == "chr3")
    assert len(train.data) + len(test.data) == len(intervals.data)


def test_chromosome_split_three_way(intervals):
    train, val, test = chromosome_split(
        intervals,
        test_chroms=["chr3"],
        val_chroms=["chr2"],
    )

    assert set(train.data["chrom"].unique()) == {"chr1"}
    assert set(val.data["chrom"].unique()) == {"chr2"}
    assert set(test.data["chrom"].unique()) == {"chr3"}

    assert _index_set(train).isdisjoint(_index_set(val))
    assert _index_set(train).isdisjoint(_index_set(test))
    assert _index_set(val).isdisjoint(_index_set(test))


def test_create_dataloader_with_seqdataset(intervals):
    dataset = SeqDataset(intervals, MockGenome(), strand_aware=False)
    loader = create_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    assert batch.shape[0] == 4
    assert batch.shape[2] == 4
    assert len(dataset) == len(intervals.data)

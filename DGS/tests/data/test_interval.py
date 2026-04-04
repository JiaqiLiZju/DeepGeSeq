"""Tests for interval operations."""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path

from DGS.Data.Interval import Interval, NamedInterval, find_overlaps, merge_intervals, find_closest

class TestIntervalOperations(unittest.TestCase):
    """Test cases for interval operations."""
    def setUp(self):
        """Set up test data."""
        # 基本区间
        self.basic_intervals = pd.DataFrame({
            'chrom': ['chr1', 'chr1', 'chr2'],
            'start': [100, 180, 150],
            'end': [150, 220, 200]
        })
        
        # 复杂区间，包含更多重叠情况
        self.complex_intervals = pd.DataFrame({
            'chrom': ['chr1', 'chr1', 'chr1', 'chr2', 'chr2'],
            'start': [100, 130, 180, 150, 190],
            'end':   [150, 160, 220, 200, 250]
        })
        
        # 带有额外列的区间
        self.annotated_intervals = pd.DataFrame({
            'chrom': ['chr1', 'chr1', 'chr2'],
            'start': [100, 180, 150],
            'end': [150, 220, 200],
            'score': [1.0, 2.0, 3.0],
            'strand': ['+', '-', '+']
        })

    def test_find_overlaps_basic(self):
        """测试基本的重叠查找"""
        query = pd.DataFrame({
            'chrom': ['chr1'],
            'start': [120],
            'end': [200]
        })
        
        result = find_overlaps(query, self.basic_intervals)
        self.assertEqual(len(result), 2)
        self.assertEqual(result['overlap_length'].tolist(), [30, 20])

    def test_find_overlaps_no_overlap(self):
        """测试无重叠情况"""
        query = pd.DataFrame({
            'chrom': ['chr1'],
            'start': [0],
            'end': [50]
        })
        
        result = find_overlaps(query, self.basic_intervals)
        self.assertEqual(len(result), 0)

    def test_find_overlaps_complete_overlap(self):
        """测试完全重叠情况"""
        query = pd.DataFrame({
            'chrom': ['chr1'],
            'start': [90],
            'end': [250]
        })
        
        result = find_overlaps(query, self.basic_intervals)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result['overlap_length'] == 
                          result['end'] - result['start']))

    def test_find_overlaps_with_annotations(self):
        """测试带注释的重叠查找"""
        query = pd.DataFrame({
            'chrom': ['chr1'],
            'start': [120],
            'end': [200],
            'name': ['test']
        })
        
        result = find_overlaps(query, self.annotated_intervals)
        self.assertTrue('query_name' in result.columns)
        self.assertTrue('score' in result.columns)
        self.assertTrue('strand' in result.columns)

    def test_merge_intervals_basic(self):
        """测试基本的区间合并"""
        result = merge_intervals(self.basic_intervals, min_distance=20)
        # chr1 两个区间间距为 30，不会在 min_distance=20 下合并
        self.assertEqual(len(result), 3)

    def test_merge_intervals_complex(self):
        """测试复杂的区间合并"""
        result = merge_intervals(self.complex_intervals, min_distance=10)
        self.assertEqual(len(result), 3)  # 应该合并chr1上的部分重叠区间

    def test_merge_intervals_no_merge(self):
        """测试不需要合并的情况"""
        result = merge_intervals(self.basic_intervals, min_distance=0)
        self.assertEqual(len(result), 3)

    def test_find_closest_basic(self):
        """测试基本的最近区间查找"""
        query = pd.DataFrame({
            'chrom': ['chr1'],
            'start': [160],
            'end': [170]
        })
        
        result = find_closest(query, self.basic_intervals, k=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result['distance'].tolist(), [10, 10])

    def test_find_closest_with_max_distance(self):
        """测试带最大距离限制的最近区间查找"""
        query = pd.DataFrame({
            'chrom': ['chr1'],
            'start': [160],
            'end': [170]
        })
        
        result = find_closest(query, self.basic_intervals, k=2, max_distance=15)
        self.assertEqual(len(result), 2)
        self.assertTrue((result['distance'] <= 15).all())

class TestInterval(unittest.TestCase):
    """Test cases for interval."""
    def setUp(self):
        """Set up test data."""
        self.basic_intervals = pd.DataFrame({
            'chrom': ['chr1', 'chr1', 'chr2'],
            'start': [100, 180, 150],
            'end': [150, 220, 200]
        })
        
        self.bed_file = Path("test.bed")
        self.bed_content = (
            "chr1\t100\t150\n"
            "chr1\t180\t220\n"
            "chr2\t150\t200\n"
        )
        with open(self.bed_file, 'w') as f:
            f.write(self.bed_content)

    def tearDown(self):
        """Clean up test files."""
        if self.bed_file.exists():
            self.bed_file.unlink()

    def test_interval_init_from_dataframe(self):
        """测试从DataFrame创建Interval"""
        interval = Interval(self.basic_intervals)
        self.assertEqual(len(interval.data), 3)
        self.assertEqual(interval.stats['total_intervals'], 3)

    def test_interval_init_from_bed(self):
        """测试从BED文件创建Interval"""
        interval = Interval(self.bed_file)
        self.assertEqual(len(interval.data), 3)
        self.assertEqual(interval.stats['total_intervals'], 3)

    def test_interval_validation_end_before_start(self):
        """测试终点在起点之前的无效区间"""
        invalid_data = pd.DataFrame({
            'chrom': ['chr1'],
            'start': [200],
            'end': [100]
        })
        
        with self.assertRaises(ValueError):
            Interval(invalid_data)

    def test_interval_validation_negative_start(self):
        """测试负起点的无效区间"""
        invalid_data = pd.DataFrame({
            'chrom': ['chr1'],
            'start': [-100],
            'end': [100]
        })
        
        with self.assertRaises(ValueError):
            Interval(invalid_data)

    def test_interval_merge_with_inner(self):
        """测试内连接合并"""
        data1 = pd.DataFrame({
            'chrom': ['chr1'],
            'start': [100],
            'end': [200]
        })
        data2 = pd.DataFrame({
            'chrom': ['chr1'],
            'start': [150],
            'end': [250]
        })
        
        interval1 = Interval(data1)
        interval2 = Interval(data2)
        
        result = interval1.merge_with(interval2, how='inner')
        self.assertEqual(len(result), 1)

    def test_interval_merge_with_outer(self):
        """测试外连接合并"""
        data1 = pd.DataFrame({
            'chrom': ['chr1'],
            'start': [100],
            'end': [200]
        })
        data2 = pd.DataFrame({
            'chrom': ['chr1'],
            'start': [300],
            'end': [400]
        })
        
        interval1 = Interval(data1)
        interval2 = Interval(data2)
        
        result = interval1.merge_with(interval2, how='outer')
        self.assertEqual(len(result), 2)

    def test_interval_merge_overlapping_with_distance(self):
        """测试带距离参数的重叠合并"""
        interval = Interval(self.basic_intervals)
        result = interval.merge_overlapping(min_distance=20)
        self.assertEqual(len(result.data), 3)

    def test_interval_find_closest_with_max_distance(self):
        """测试带最大距离的最近区间查找"""
        interval = Interval(self.basic_intervals)
        query = pd.DataFrame({
            'chrom': ['chr1'],
            'start': [160],
            'end': [170]
        })
        
        result = interval.find_closest(query, k=1, max_distance=15)
        # 当前方法语义是“以 self 为 query”，因此 chr1 的两个区间都会返回
        self.assertEqual(len(result), 2)
        self.assertTrue((result['distance'] <= 15).all())

    def test_interval_to_bed_with_extra_columns(self):
        """测试带额外列的BED文件输出"""
        data = pd.DataFrame({
            'chrom': ['chr1'],
            'start': [100],
            'end': [200],
            'name': ['test'],
            'score': [1.0],
            'strand': ['+']
        })
        
        interval = Interval(data)
        output_file = Path("test_extra.bed")
        interval.to_bed(output_file)
        
        self.assertTrue(output_file.exists())
        output_file.unlink()

class TestNamedInterval(unittest.TestCase):
    """Test cases for named interval."""
    def setUp(self):
        """Set up test data."""
        self.named_intervals = pd.DataFrame({
            'chrom': ['chr1', 'chr1', 'chr2'],
            'start': [100, 180, 150],
            'end': [150, 220, 200],
            'name': ['A', 'B', 'C']
        })

    def test_named_interval_init_with_names(self):
        """测试带名称初始化"""
        interval = NamedInterval(self.named_intervals)
        self.assertEqual(len(interval.data), 3)
        self.assertIn('name', interval.data.columns)
        self.assertEqual(interval.data['name'].tolist(), ['A', 'B', 'C'])

    def test_named_interval_init_without_names(self):
        """测试无名称初始化"""
        data = self.named_intervals.drop('name', axis=1)
        interval = NamedInterval(data)
        self.assertTrue(all(interval.data[interval.name_col].str.startswith('interval_')))

    def test_named_interval_auto_names_with_prefix(self):
        """测试自定义前缀的自动命名"""
        data = self.named_intervals.drop('name', axis=1)
        interval = NamedInterval(data, name_prefix='test_')
        self.assertTrue(all(interval.data[interval.name_col].str.startswith('test_')))

    def test_named_interval_duplicate_names_handling(self):
        """测试重复名称处理"""
        data = pd.DataFrame({
            'chrom': ['chr1', 'chr1', 'chr1'],
            'start': [100, 150, 200],
            'end': [120, 170, 220],
            'name': ['A', 'A', 'A']
        })
        
        interval = NamedInterval(data)
        names = interval.data[interval.name_col].tolist()
        self.assertEqual(len(set(names)), 3)
        self.assertEqual(names, ['A', 'A_1', 'A_2'])

    def test_named_interval_merge_with_name_handling(self):
        """测试合并时的名称处理"""
        data1 = pd.DataFrame({
            'chrom': ['chr1'],
            'start': [100],
            'end': [200],
            'name': ['A']
        })
        data2 = pd.DataFrame({
            'chrom': ['chr1'],
            'start': [150],
            'end': [250],
            'name': ['B']
        })
        
        interval1 = NamedInterval(data1)
        interval2 = NamedInterval(data2)
        
        result = interval1.merge_with(interval2)
        self.assertEqual(len(result), 1)
        self.assertIn('query_name', result.columns)
        self.assertEqual(result['query_name'].iloc[0], 'A')

    def test_named_interval_copy_with_names(self):
        """测试带名称的深拷贝"""
        interval = NamedInterval(self.named_intervals)
        copied = interval.copy()
        
        self.assertNotEqual(id(interval.data), id(copied.data))
        self.assertTrue(interval.data.equals(copied.data))
        self.assertEqual(
            interval.data[interval.name_col].tolist(),
            copied.data[copied.name_col].tolist()
        )

if __name__ == '__main__':
    unittest.main() 

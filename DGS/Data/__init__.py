"""
DGS Data Processing Module

This module provides comprehensive tools for genomic data processing and management:

Core Components:
1. Interval Operations (Interval.py):
   - Genomic interval manipulation
   - Overlap detection and merging
   - Distance calculations
   - Statistical analysis

2. Sequence Processing (Sequence.py):
   - DNA sequence manipulation
   - One-hot encoding/decoding
   - Sequence complexity metrics
   - FASTA file integration

3. Target Data Management (Target.py):
   - Multi-task target handling
   - BED and BigWig support
   - Data encoding and statistics
   - Task configuration

4. Dataset Classes (Dataset.py):
   - PyTorch dataset implementations
   - Sequence extraction
   - Batch processing
   - Multi-task learning support

5. Data Sampling (Sampler.py):
   - Train/test splitting
   - Chromosome-based splitting
   - Random sampling
   - Cross-validation utilities

The module is designed for efficient processing of genomic data in deep learning
applications, with particular focus on sequence analysis tasks.
"""

from .Interval import *
from .Sequence import *
from .Target import *
from .Sampler import *
from .Dataset import *
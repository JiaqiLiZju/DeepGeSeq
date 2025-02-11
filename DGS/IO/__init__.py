"""Input/Output module for genomic data processing in DGS.

This module provides a unified interface for reading and processing various genomic data formats:

File Formats
-----------
BED/narrowPeak
    Genomic interval data with optional metadata:
    - Standard BED format (3-12 columns)
    - ENCODE narrowPeak format (10 columns)
    - Automatic format detection and validation

FASTA
    DNA sequence data with random access:
    - Efficient sequence extraction by coordinates
    - Support for indexed FASTA files
    - Batch processing capabilities
    - Automatic index creation

BigWig
    Continuous genomic signal data:
    - Random access to genomic regions
    - Signal aggregation functions
    - Multi-track support
    - Flexible binning options

Classes
-------
FastaReader
    Reader for FASTA sequence files with random access and batch processing

BedReader
    Reader for BED and BED-like interval files with format validation

BigWigReader
    Reader for BigWig signal files with aggregation capabilities

Notes
-----
All readers follow a consistent interface pattern:
1. File validation on initialization
2. Standardized interval data format
3. Comprehensive error handling
4. Debug-level logging
"""

import logging

# Configure module logger
logger = logging.getLogger("dgs.io")
logger.addHandler(logging.NullHandler())  # Prevent no handler warning

# Import readers
from .fasta import FastaReader
from .bed import BedReader
from .bigwig import BigWigReader

__all__ = [
    "FastaReader",  # DNA sequence reading
    "BedReader",    # Genomic interval reading
    "BigWigReader"  # Signal data reading
]

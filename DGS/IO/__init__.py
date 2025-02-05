"""IO module for DGS.

Provides unified interfaces for reading genomic data files:
- BED/narrowPeak: Genomic intervals
- FASTA: DNA sequences
- BigWig: Genomic signal tracks
"""

import logging

# Configure logger
logger = logging.getLogger("dgs.io")

from .fasta import FastaReader
from .bed import BedReader
from .bigwig import BigWigReader

__all__ = [
    "FastaReader",
    "BedReader", 
    "BigWigReader"
]

"""VCF file reader module for genomic variants.

This module provides functionality for reading and processing Variant Call Format (VCF)
files, which store gene sequence variations (SNPs, indels, etc).

Functions
---------
read_vcf(filename)
    Read a VCF file into a pandas DataFrame with standardized columns

Notes
-----
The module currently supports basic VCF parsing with the following features:
- Standard VCF columns (CHROM, POS, ID, etc.)
- Comment filtering
- Basic data type conversion
- Memory-efficient reading
"""

import pandas as pd
import logging

from . import logger

def read_vcf(filename: str) -> pd.DataFrame:
    """Read a VCF file into a pandas DataFrame.

    Parameters
    ----------
    filename : str
        Path to the VCF file to read

    Returns
    -------
    pd.DataFrame
        DataFrame containing VCF records with columns:
        - CHROM: Chromosome name (str)
        - POS: Position (int)
        - ID: Variant identifier (str)
        - REF: Reference allele (str)
        - ALT: Alternate allele(s) (str)
        - QUAL: Quality score (str)
        - FILTER: Filter status (str)
        - INFO: Additional information (str)
        - FORMAT: Genotype format (str)

    Notes
    -----
    - Comments (lines starting with #) are automatically filtered
    - Only the first 9 standard VCF columns are read
    - All columns except POS are read as strings
    - Missing values are preserved as is
    """
    logger.debug(f"Reading VCF file: {filename}")

    # Define column names and types
    names = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    dtypes = {name: str for name in names}
    dtypes['POS'] = int

    try:
        # Read VCF file
        vcf = pd.read_csv(
            filename,
            delimiter='\t',
            comment='#',
            names=names,
            dtype=dtypes,
            usecols=range(9)
        )
        
        logger.debug(f"Successfully read {len(vcf)} variants")
        return vcf
        
    except Exception as e:
        logger.error(f"Failed to read VCF file {filename}: {str(e)}")
        raise


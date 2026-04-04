"""VCF parsing helpers used by prediction workflows.

Purpose:
    Parse core VCF columns into a pandas DataFrame for downstream processing.

Main Responsibilities:
    - Read standard VCF records while skipping metadata/comment lines.
    - Enforce stable column naming for `CHROM`, `POS`, `REF`, `ALT`, and peers.
    - Provide lightweight logging for success and failure cases.

Key Runtime Notes:
    - Only the first 9 standard VCF columns are loaded.
    - `POS` is parsed as integer and other columns are parsed as strings.
    - Parsing errors are logged and re-raised.
"""

import pandas as pd

from . import logger

def read_vcf(filename: str) -> pd.DataFrame:
    """Read a VCF file into a pandas DataFrame.

    Args:
    filename : str
        Path to the VCF file to read

    Returns:
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

    Notes:
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

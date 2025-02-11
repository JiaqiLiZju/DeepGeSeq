"""
Variant Effect Prediction Module

This module provides tools for predicting the functional effects of genetic variants
using deep learning models. It implements a comprehensive framework for:
- Processing VCF files and genomic sequences
- Generating variant-centered sequence windows
- Computing variant effect scores
- Batch prediction and analysis

Key Components:
1. Data Processing:
   - VCF file parsing and validation
   - Sequence extraction and mutation
   - Variant context generation
   - Batch processing utilities

2. Prediction Pipeline:
   - Model inference setup
   - Variant effect scoring
   - Multi-task prediction handling
   - Result aggregation

3. Output Generation:
   - Prediction score calculation
   - Result formatting and export
   - Statistical analysis
   - Visualization support

The module is designed to handle various types of genetic variants including:
- Single nucleotide polymorphisms (SNPs)
- Insertions and deletions (indels)
- Multiple nucleotide variants (MNVs)
"""

import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Union, Optional
from pathlib import Path

def read_vcf(filename) -> pd.DataFrame:
    """
    Parse and validate a VCF file containing genetic variants.

    Args:
        filename (str): Path to the VCF file

    Returns:
        pd.DataFrame: Parsed variant information with columns:
            - CHROM: Chromosome name
            - POS: 1-based variant position
            - ID: Variant identifier
            - REF: Reference allele
            - ALT: Alternative allele
            - QUAL: Quality score
            - FILTER: Filter status
            - INFO: Additional information
            - FORMAT: Genotype format

    Note:
        The function handles standard VCF format (v4.0+) and
        automatically detects column types.
    """
    names = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    dtypes = {name: str for name in names}
    dtypes['POS'] = int

    variants = pd.read_csv(filename, delimiter='\t', comment='#', names=names, 
        dtype=dtypes, usecols=range(9))
    return variants


from ..Data.Interval import Interval
def variants_to_intervals(variants, seq_len=1000) -> Interval:
    """
    Convert variant positions to sequence intervals for analysis.

    This function creates fixed-length sequence windows centered on each variant
    for model input preparation.

    Args:
        variants (pd.DataFrame): DataFrame containing variant information
        seq_len (int): Length of sequence context around variants

    Returns:
        Interval: Interval object containing sequence windows
            Each interval includes:
            - Chromosome name
            - Start position (0-based)
            - End position (exclusive)

    Note:
        The function automatically handles coordinate conversion
        between 1-based VCF and 0-based interval formats.
    """
    intervals = []
    for _, mutation in variants.iterrows():
        chrom = mutation['CHROM']
        pos = mutation['POS'] - 1 #VCF is 1-based
        mid = seq_len // 2
        start = pos - mid
        end = seq_len + start
        intervals.append({'chrom': chrom, 'start': start, 'end': end})
    intervals = pd.DataFrame(intervals)
    intervals = Interval(intervals)
    return intervals


# TODO: move to Data.Sequence.DNASeq
def mutate(seq, var_ref, var_alt):
    """
    Apply genetic variants to a reference sequence.

    This function handles different types of variants:
    - SNPs: Direct base substitution
    - Deletions: Remove bases and pad with N's
    - Insertions: Add bases and trim to maintain length

    Args:
        seq (str): Reference sequence
        var_ref (str): Reference allele
        var_alt (str): Alternative allele

    Returns:
        str: Mutated sequence of the same length as input

    Note:
        The function maintains sequence length by appropriate
        padding or trimming based on variant type.
    """
    mid = len(seq) // 2
    target_len = len(seq)

    if len(var_ref) == len(var_alt): # SNP
        seq = seq[:mid] + var_alt + seq[mid+len(var_ref):]

    elif len(var_ref) > len(var_alt): # deletion
        pad = 'N' * (len(var_ref) - len(var_alt))
        seq = seq[:mid] + var_alt + seq[mid+len(var_ref):] + pad

    else: # insertion
        trim = (len(var_alt) - len(var_ref)) // 2
        seq = seq[:mid] + var_alt + seq[mid+len(var_ref):]
        seq = seq[trim:trim+target_len]

    return seq

from ..Data.Dataset import SeqDataset
from ..Data.Sequence import DNASeq
class VariantDataset(SeqDataset):
    """
    Dataset class for variant effect prediction.

    This class manages sequence data for variant analysis:
    - Loads reference and variant sequences
    - Validates variant positions
    - Generates paired sequences for comparison

    Attributes:
        genome: Reference genome object
        variants (pd.DataFrame): Variant information
        var_ref (List[str]): Reference alleles
        var_alt (List[str]): Alternative alleles

    Methods:
        check_reference: Validate reference sequence matches
        __getitem__: Get reference-variant sequence pairs
    """
    
    def __init__(self, genome, variants, target_len=1000, **kwargs):
        """
        Initialize variant dataset.

        Args:
            genome: Reference genome object
            variants (pd.DataFrame): Variant information
            target_len (int): Sequence context length
            **kwargs: Additional arguments for parent class
        """
        intervals = variants_to_intervals(variants, seq_len=target_len)
        super().__init__(intervals, genome, **kwargs)
        self.variants = variants
        self.var_ref = variants['REF'].tolist()
        self.var_alt = variants['ALT'].tolist()
    
    def check_reference(self, seq, ref):
        """
        Validate that extracted sequence matches reference allele.

        Args:
            seq (str): Extracted sequence
            ref (str): Reference allele from VCF

        Returns:
            bool: True if reference matches, False otherwise
        """
        start_idx = len(seq) // 2 # variant-centred
        stop_idx = start_idx + len(ref)
        return seq[start_idx:stop_idx] == ref

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        seq_ref = self.seqs[idx]
        seq_alt = self.seqs[idx]

        var_ref = self.var_ref[idx]
        var_alt = self.var_alt[idx]

        if self.check_reference(seq_ref.sequence, var_ref):
            seq_alt = mutate(seq_alt.sequence, var_alt, var_ref)
            seq_alt = DNASeq(seq_alt)
        else:
            variant = self.variants.iloc[idx]
            raise ValueError(f"Reference sequence does not match the reference allele of the current variant at the current position: {variant}")

        return seq_ref.to_onehot(), seq_alt.to_onehot()


from ..Data.Sequence import one_hot_encode
def variant_effect_prediction(model, X_ref, X_alt, device=torch.device('cpu')):
    """
    Predict the effect of genetic variants using a trained model.

    This function:
    1. Processes reference and alternative sequences
    2. Runs model predictions
    3. Returns prediction scores for comparison

    Args:
        model (nn.Module): Trained neural network model
        X_ref: Reference sequences (str, array, or tensor)
        X_alt: Alternative sequences (str, array, or tensor)
        device (str): Computation device ('cuda' or 'cpu')

    Returns:
        tuple: (reference_predictions, alternate_predictions)
            Both in numpy array format with shape (N, num_tasks)

    Note:
        The function automatically handles different input formats
        and ensures consistent tensor shapes.
    """
    if type(X_ref) == str:
        X_ref = one_hot_encode(X_ref)
    if type(X_alt) == str:
        X_alt = one_hot_encode(X_alt)

    if X_ref.ndim == 2:
        X_ref = X_ref[None, :, :]
    if X_alt.ndim == 2:
        X_alt = X_alt[None, :, :]

    if X_ref.shape[-1] != 4:
        X_ref = X_ref.swapaxes(1,-1)
    if X_alt.shape[-1] != 4:
        X_alt = X_alt.swapaxes(1,-1)

    # Set model to evaluation mode
    model.eval()
    model.to(device)

    p_ref = None
    with torch.no_grad():
        d = torch.from_numpy(X_ref.astype(np.float32)).to(device)
        e = model.forward(d)#, return_only_embeddings=True)
        p_ref = e.data.cpu().numpy()
    #print(p_ref.shape)

    p_alt = None
    with torch.no_grad():
        d_alt = torch.from_numpy(X_alt.astype(np.float32)).to(device)
        e_alt = model.forward(d_alt)#, return_only_embeddings=True)
        p_alt = e_alt.data.cpu().numpy()
    #print(p_alt.shape)

    return p_ref, p_alt


def metric_predicted_effect(p_ref, p_alt, metric_func='diff', mean_by_tasks=True):
    """
    Calculate variant effect scores using specified metrics.

    Supported metrics:
    - diff: Absolute difference between predictions
    - ratio: Ratio of predicted effects
    - log_ratio: Log ratio of predicted effects
    - max: Maximum predicted effect
    - min: Minimum predicted effect
    - custom: User-provided callable function

    Args:
        p_ref (np.ndarray): Reference sequence predictions
        p_alt (np.ndarray): Alternative sequence predictions
        metric_func (str or callable): Metric calculation method
        mean_by_tasks (bool): Whether to average across tasks

    Returns:
        np.ndarray: Variant effect scores

    Note:
        For multi-task models, scores can be averaged or kept separate
        based on mean_by_tasks parameter.
    """
    if metric_func == 'diff':
        p_eff = p_alt - p_ref
        p_eff = np.abs(p_eff)

    elif metric_func == 'ratio':
        p_eff = p_alt / p_ref

    elif metric_func == 'log_ratio':
        p_eff = np.log(p_alt / p_ref)

    elif metric_func == 'max':
        p_eff = np.max(p_alt, axis=1) - np.max(p_ref, axis=1)

    elif metric_func == 'min':
        p_eff = np.min(p_alt, axis=1) - np.min(p_ref, axis=1)

    elif callable(metric_func):
        p_eff = metric_func(p_ref, p_alt)

    else:
        raise ValueError(f"Invalid metric function: {metric_func}. Must be one of: {allowed_metrics}")

    if mean_by_tasks:
        p_eff = p_eff.mean(axis=1)

    return p_eff


def vep_centred_on_ds(model, ds,
                metric_func='diff', mean_by_tasks=True, 
                device=torch.device('cpu')):
    """
    Perform variant effect prediction on a complete dataset.

    This function:
    1. Processes all variants in the dataset
    2. Calculates effect scores using specified metric
    3. Aggregates results across all variants

    Args:
        model (nn.Module): Trained neural network model
        ds (VariantDataset): Dataset containing variants
        metric_func (str or callable): Effect score calculation method
        mean_by_tasks (bool): Whether to average across tasks
        device (str): Computation device ('cuda' or 'cpu')

    Returns:
        np.ndarray: Effect scores for all variants
    """
    P_diff = []
    for i in range(len(ds)):
        seq_ref, seq_alt = ds[i]
        p_ref, p_alt = variant_effect_prediction(model, seq_ref, seq_alt, device=device)
        p_eff = metric_predicted_effect(p_ref, p_alt, metric_func, mean_by_tasks)
        P_diff.append(p_eff)

    P_diff = np.concatenate(P_diff, axis=0)
    return P_diff


def vep_centred_from_files(model, genome_filename, bcf_filename, 
                           target_len=1000, device=torch.device('cpu'), 
                           metric_func='diff', mean_by_tasks=True, 
                           return_df=True, save_path=None):
    """
    Complete variant effect prediction pipeline from input files.

    This function provides a high-level interface for:
    1. Loading genome and variant data
    2. Setting up prediction pipeline
    3. Calculating variant effects
    4. Saving and returning results

    Args:
        model (nn.Module): Trained neural network model
        genome_filename (str): Path to reference genome
        bcf_filename (str): Path to VCF/BCF file
        target_len (int): Sequence context length
        device (str): Computation device
        metric_func (str or callable): Effect score method
        mean_by_tasks (bool): Whether to average across tasks
        return_df (bool): Whether to return DataFrame
        save_path (str, optional): Path to save results

    Returns:
        Union[pd.DataFrame, Tuple[np.ndarray, pd.DataFrame]]:
            - If return_df=True: DataFrame with predictions
            - If return_df=False: (predictions, variant_info)

    Note:
        Results can be saved to disk and/or returned for
        further analysis.
    """
    variant_df = read_vcf(bcf_filename)
    ds = VariantDataset(genome_filename, variant_df, target_len=target_len)
    
    P_diff = vep_centred_on_ds(model, ds, metric_func, mean_by_tasks, device)

    if save_path:
        variant_df.to_csv(os.path.join(save_path, 'variant_df.csv'), index=False)
        np.save(os.path.join(save_path, 'P_diff.npy'), P_diff)

    if return_df:
        variant_df['P_diff'] = P_diff
        return variant_df
    else:
        return P_diff, variant_df
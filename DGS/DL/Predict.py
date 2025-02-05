import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Union, Optional
from pathlib import Path

def read_vcf(filename) -> pd.DataFrame:
	names = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", 
		"FORMAT"]
	dtypes = {name: str for name in names}
	dtypes['POS'] = int

	variants = pd.read_csv(filename, delimiter='\t', comment='#', names=names, 
		dtype=dtypes, usecols=range(9))
	return variants


from ..Data.Interval import Interval
def variants_to_intervals(variants, seq_len=1000) -> Interval:
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
    """Dataset for variants."""
    
    def __init__(self, genome, variants, target_len=1000, **kwargs):
        intervals = variants_to_intervals(variants, seq_len=target_len)
        super().__init__(intervals, genome, **kwargs)
        self.variants = variants
        self.var_ref = variants['REF'].tolist()
        self.var_alt = variants['ALT'].tolist()
    
    def check_reference(self, seq, ref):
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
    Predict the effect of a variant on a sequence.
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
    """handle predicted effect distance metric and mean by tasks
    Supported metrics:
        - diff: absolute difference
        - ratio: ratio of predicted effects
        - log_ratio: log ratio of predicted effects
        - max: maximum predicted effect
        - min: minimum predicted effect
        - callable
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
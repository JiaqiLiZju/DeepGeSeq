"""Attribution and motif analysis helpers for trained models.

Purpose:
    Generate sequence attributions and downstream motif/seqlet summaries.

Main Responsibilities:
    - Compute DeepLIFT/SHAP-style attributions for tensors and datasets.
    - Export attribution artifacts for TF-MoDISco-lite workflows.
    - Run optional seqlet calling and motif annotation pipelines.

Key Runtime Notes:
    - Requires `tangermeme` for attribution and seqlet operations.
    - Motif enrichment/report generation additionally requires `modisco` CLI.
    - Batched attribution mode is available through `batch_size` parameters.
"""

import os
import logging
import subprocess
import shutil
from typing import Optional

import torch
import numpy as np
from torch.utils.data import DataLoader

try:
    from tangermeme.deep_lift_shap import deep_lift_shap
    from tangermeme.seqlet import recursive_seqlets
    from tangermeme.annotate import annotate_seqlets
    from tangermeme.io import read_meme
    _TANGERMEME_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised in dependency-missing envs
    deep_lift_shap = None
    recursive_seqlets = None
    annotate_seqlets = None
    read_meme = None
    _TANGERMEME_IMPORT_ERROR = exc

logger = logging.getLogger("dgs.explain")


def _ensure_explain_dependencies(require_modisco: bool = False) -> None:
    """Raise clear runtime errors for optional explain dependencies."""
    if _TANGERMEME_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Explain mode requires optional dependency 'tangermeme'. "
            "Install it with `pip install tangermeme`."
        ) from _TANGERMEME_IMPORT_ERROR
    if require_modisco and shutil.which("modisco") is None:
        raise RuntimeError(
            "Explain mode requires the `modisco` CLI in PATH for motif workflows."
        )

def calculate_shap(model, X, target, device):
    """
    Calculate SHAP (SHapley Additive exPlanations) attributions for model predictions.

    This function computes importance scores for each position in the input sequences
    using the DeepLIFT algorithm adapted for SHAP values.

    Args:
        model (nn.Module): Trained neural network model
        X (torch.Tensor): Input sequences in one-hot encoded format (N, 4, L)
        target (int): Target task index for multi-task models
        device (str): Computation device ('cuda' or 'cpu')

    Returns:
        np.ndarray: Attribution scores with shape matching input (N, 4, L)

    Note:
        The function automatically handles device placement and fallback behavior.
        If attribution computation fails, an error is printed and zero-valued
        attributions are returned with the same shape as input.
    """

    _ensure_explain_dependencies()

    # Set model to evaluation mode
    model.eval()
    model.to(device)
    X = X.to(device)

    try:
        # Calculate DeepLIFT/SHAP attributions
        X_attr = deep_lift_shap(model, X, target=target)
        X_attr = X_attr.cpu().numpy()
    except Exception as e:
        logger.error("Error calculating SHAP attributions: %s", e)
        X_attr = np.zeros_like(X.cpu().numpy())
    
    return X_attr

def _to_batch_input(data) -> torch.Tensor:
    """Convert sample/batch data to (N, 4, L) float tensor."""
    if isinstance(data, (tuple, list)):
        data = data[0]

    if isinstance(data, np.ndarray):
        x = torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        x = data
    else:
        x = torch.tensor(data)

    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.shape[1] != 4:
        x = x.transpose(1, 2)
    return x.float()


def calculate_shap_on_ds(model, ds, target, device, batch_size: Optional[int] = None):
    """
    Calculate SHAP attributions for an entire dataset.

    This function processes a dataset in batches, handling various input formats
    and ensuring consistent tensor shapes.

    Args:
        model (nn.Module): Trained neural network model
        ds (Dataset): Dataset containing sequences
        target (int): Target task index for multi-task models
        device (str): Computation device ('cuda' or 'cpu')
        batch_size (int, optional): Batch size for attribution inference.
            If omitted or <=1, samples are processed one-by-one.

    Returns:
        tuple: (sequences, attributions)
            - sequences: Original sequences in one-hot format (N, 4, L)
            - attributions: SHAP values for each sequence (N, 4, L)
    """
    _ensure_explain_dependencies()

    X, X_attr = [], []
    if not batch_size or batch_size <= 1:
        for i in range(len(ds)):
            x = _to_batch_input(ds[i])
            x_attr = calculate_shap(model, x, target, device)
            X.append(x.cpu().numpy())
            X_attr.append(x_attr)
    else:
        logger.info("Using batched SHAP calculation with batch_size=%s", batch_size)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        for batch in dataloader:
            x = _to_batch_input(batch)
            x_attr = calculate_shap(model, x, target, device)
            X.append(x.cpu().numpy())
            X_attr.append(x_attr)

    X = np.concatenate(X, axis=0)
    X_attr = np.concatenate(X_attr, axis=0)

    return X, X_attr


def motif_enrich(
    model,
    ds,
    target,
    output_dir="motif_results",
    max_seqlets=2000,
    device=torch.device("cpu"),
    batch_size: Optional[int] = None,
):
    """
    Perform comprehensive motif enrichment analysis using model interpretations.

    This function:
    1. Calculates SHAP attributions for input sequences
    2. Identifies important sequence patterns
    3. Runs TF-MoDISco-lite for motif discovery
    4. Generates visualization reports

    Args:
        model (nn.Module): Trained neural network model
        ds (Dataset): Dataset containing sequences
        target (int): Target task index for multi-task models
        output_dir (str): Directory to save analysis results
        max_seqlets (int): Maximum number of sequence elements to analyze
        device (str): Computation device ('cuda' or 'cpu')
        batch_size (int, optional): Batch size for SHAP inference.

    Returns:
        str: Path to generated motifs file

    Raises:
        subprocess.CalledProcessError:
            If `modisco motifs` or `modisco report` command fails.

    Note:
        Runtime requirements:
        - `tangermeme` must be importable.
        - `modisco` command must be available in PATH.

        Results include:
        - Sequence attributions (NPZ format)
        - Discovered motifs (MEME format)
        - Visualization reports (HTML/PDF)
        - Raw data for further analysis
    """
    
    _ensure_explain_dependencies(require_modisco=True)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
        
    # Calculate DeepLIFT/SHAP attributions
    logger.info("Calculating DeepLIFT/SHAP attributions...")
    X, X_attr = calculate_shap_on_ds(model, ds, target=target, device=device, batch_size=batch_size)
    
    # Save one-hot encoded sequences and attributions
    logger.info("Saving sequences and attributions...")
    ohe_path = os.path.join(output_dir, "ohe.npz")
    shap_path = os.path.join(output_dir, "shap.npz")
    
    np.savez_compressed(ohe_path, X)
    np.savez_compressed(shap_path, X_attr)
    
    # Run TF-MoDISco-lite
    logger.info("Running TF-MoDISco-lite...")
    modisco_output = os.path.join(output_dir, "modisco_results.h5")
    motifs_output = os.path.join(output_dir, "motifs.txt")
    
    # Run modisco motifs command
    cmd = f"modisco motifs -s {ohe_path} -a {shap_path} -n {max_seqlets} -o {modisco_output}"
    subprocess.run(cmd, shell=True, check=True)
    
    # Generate report and motifs.txt
    cmd = f"modisco report -i {modisco_output} -o {output_dir} -s {output_dir}"
    subprocess.run(cmd, shell=True, check=True)
    
    logger.info(f"Motif analysis complete. Results saved in {output_dir}")
    return motifs_output

def Seqlet_Calling(model, ds, target, output_dir="seqlet_results", motif_db=None, device=torch.device("cpu")):
    """
    Identify and annotate regulatory elements (seqlets) in sequences.

    This function performs:
    1. Attribution calculation for sequences
    2. Seqlet identification using recursive algorithm
    3. Motif annotation if database provided
    4. Statistical significance assessment

    Args:
        model (nn.Module): Trained neural network model
        ds (Dataset): Dataset containing sequences
        target (int): Target task index for multi-task models
        output_dir (str): Directory to save results
        motif_db (str, optional): Path to MEME format motif database
        device (str): Computation device ('cuda' or 'cpu')

    Returns:
        DataFrame: Identified seqlets with annotations
            Columns include:
            - Sequence coordinates
            - Importance scores
            - Motif matches (if database provided)
            - Statistical significance

    Raises:
        FileNotFoundError:
            Potentially raised by downstream motif readers if motif files are missing.

    Note:
        Runtime requirements:
        - `tangermeme` must be importable.
        - If motif annotation is requested, `motif_db` should point to a valid
          MEME-format motif file.

        Results are saved in BED format for compatibility with
        genome browsers and downstream analysis tools.
    """
    
    _ensure_explain_dependencies()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate DeepLIFT/SHAP attributions
    logger.info("Calculating DeepLIFT/SHAP attributions...")
    X, X_attr = calculate_shap_on_ds(model, ds, target=target, device=device)
    
    # Call seqlets using recursive algorithm
    logger.info("Calling seqlets...")
    seqlets = recursive_seqlets(np.sum(X_attr, axis=1))  # Sum across channels for overall importance
    
    # Save seqlets information
    seqlets.to_csv(os.path.join(output_dir, "seqlets.bed"), sep="\t", header=False, index=False)
    
    # Annotate seqlets if motif database is provided
    if motif_db is not None and os.path.exists(motif_db):
        logger.info("Annotating seqlets with motif database...")
        motifs = read_meme(motif_db)
        motif_idxs, motif_pvalues = annotate_seqlets(X, seqlets, motifs)
        
        # Save annotation results
        seqlets['motif_indices'] = motif_idxs
        seqlets['motif_pvalues'] = motif_pvalues
        seqlets.to_csv(os.path.join(output_dir, "seqlets.bed"), sep="\t", header=False, index=False)
    
    logger.info(f"Seqlet analysis complete. Results saved in {output_dir}")
    return seqlets

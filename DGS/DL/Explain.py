from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.seqlet import recursive_seqlets
from tangermeme.annotate import annotate_seqlets
from tangermeme.io import read_meme

import os
import logging
import subprocess

import torch
import numpy as np

logger = logging.getLogger("dgs.explain")

def calculate_shap(model, X, target, device):
    """
    Calculate SHAP attributions for a given model and input sequences.
    """

    # Set model to evaluation mode
    model.eval()
    model.to(device)
    X = X.to(device)

    try:
        # Calculate DeepLIFT/SHAP attributions
        X_attr = deep_lift_shap(model, X, target=target)
        X_attr = X_attr.cpu().numpy()
    except Exception as e:
        print(f"Error calculating SHAP attributions: {e}")
        X_attr = np.zeros_like(X.cpu().numpy())
    
    return X_attr

def calculate_shap_on_ds(model, ds, target, device):
    """
    Calculate SHAP attributions for a given model and input sequences.
    """

    X, X_attr = [], []
    for i in range(len(ds)):
        data = ds[i]
        if len(data) > 1:
            x = data[0]
        else:
            x = data

        # Convert X to (N, 4, L) format if needed
        if x.ndim == 2:
            x = x[None, ...]

        if x.shape[1] != 4:
            x = x.swapaxes(1,-1)

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        x_attr = calculate_shap(model, x, target, device)
        
        X.append(x.cpu().numpy())
        X_attr.append(x_attr)

    X = np.concatenate(X, axis=0)
    X_attr = np.concatenate(X_attr, axis=0)

    return X, X_attr


def motif_enrich(model, ds, target, output_dir="motif_results", max_seqlets=2000, device=torch.device("cpu")):
    """
    Perform motif enrichment analysis using TangerMeme and TF-MoDISco-lite
    
    Args:
        model: PyTorch model
        X: Input sequences tensor (N, 4, L) or (N, L, 4)
        target: Target task index for multi-task models
        output_dir: Directory to save results
        max_seqlets: Maximum number of seqlets per metacluster
        
    Returns:
        str: Path to generated motifs.txt file
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
        
    # Calculate DeepLIFT/SHAP attributions
    logger.info("Calculating DeepLIFT/SHAP attributions...")
    X, X_attr = calculate_shap_on_ds(model, ds, target=target, device=device)
    
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
    Perform seqlet calling and annotation using TangerMeme
    
    Args:
        model: PyTorch model
        X: Input sequences tensor (N, 4, L) or (N, L, 4)
        target: Target task index for multi-task models
        output_dir: Directory to save results
        motif_db: Path to MEME format motif database for annotation
        
    Returns:
        dict: Dictionary containing seqlets and their annotations
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate DeepLIFT/SHAP attributions
    logger.info("Calculating DeepLIFT/SHAP attributions...")
    X, X_attr = calculate_shap_on_ds(model, ds, target=target, device=device)
    
    # Call seqlets using recursive algorithm
    logger.info("Calling seqlets...")
    seqlets = recursive_seqlets(X_attr.sum(dim=1))  # Sum across channels for overall importance
    
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

"""DGS: Deep Learning Toolkit for Genomics

This package provides tools for applying deep learning to genomics data.
Main features:
- Deep learning model training and evaluation 
- Model interpretation and visualization
- Variant effect prediction
- Hyperparameter optimization
"""

import sys
import datetime
import logging
from pathlib import Path

import torch
import random
import numpy as np

# Version control
__version__ = "0.1.0"
__author__ = "Jiaqili@zju.edu.cn"
__email__ = "Jiaqili@zju.edu.cn"


# setup functions
def _set_random_seed(seed: int = 12):
    """Set Python and NumPy random seeds"""
    random.seed(seed)
    np.random.seed(seed)

def _set_torch_seed(seed: int = 12):
    """Set PyTorch random seeds"""
    torch.random.seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _set_torch_benchmark():
    """Setup PyTorch benchmark mode"""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def _get_device(gpu_id: int = 0) -> torch.device:
    if not torch.cuda.is_available() or gpu_id < 0:
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu_id}")

def initialize_logger(output_path: str, verbosity: int = 1):
    """Initialize DGS logging system"""
    logger = logging.getLogger("dgs")
    if logger.handlers:
        return logger

    level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logger.setLevel(level_map.get(verbosity, logging.INFO))

    # File handler
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(output_path)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler 
    console_formatter = logging.Formatter("%(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def setup_environment(output_dir: str, 
                      verbose: int = 1, 
                      seed: int = 12, 
                      benchmark: bool = True,
                      gpu: int = 0) -> tuple[torch.device, logging.Logger]:
    """Setup execution environment"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = str(output_dir / f"{output_dir}_{timestamp}.log")
    initialize_logger(
        output_path=log_file,
        verbosity=verbose
    )
    logger = logging.getLogger("dgs")
    
    # Set random seeds
    _set_random_seed(seed)
    _set_torch_seed(seed)
    logger.debug(f"Random seed set to {seed}")

    # Setup benchmark
    if benchmark:
        _set_torch_benchmark()
        logger.info("PyTorch benchmark mode enabled")

    # Setup device
    device = _get_device(gpu)
    logger.info(f"Using device: {device}")
        
    return device, logger

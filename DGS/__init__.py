"""DGS package bootstrap utilities.

This module defines package metadata and shared runtime setup helpers used by
the CLI entrypoint and programmatic workflows. It centralizes environment
configuration for reproducibility, logging, and device selection.

Main responsibilities:
    - Set random seeds for Python, NumPy, and PyTorch.
    - Configure cuDNN speed vs. determinism behavior.
    - Initialize package-scoped logging handlers.
    - Create output directories and choose compute devices.
"""

import sys
import datetime
import logging
from pathlib import Path
from typing import Tuple

import random
import numpy as np
import torch

# Version control
__version__ = "0.1.0"
__author__ = "Jiaqili@zju.edu.cn"
__email__ = "Jiaqili@zju.edu.cn"


# setup functions
def _set_random_seed(seed: int = 12):
    """Set Python and NumPy random seeds.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

def _set_torch_seed(seed: int = 12):
    """Set PyTorch random seeds for CPU and CUDA backends.

    Args:
        seed: Random seed value.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def _set_torch_backend(benchmark: bool = True):
    """Configure cuDNN backend flags in a consistent way.

    Benchmark mode accelerates fixed-shape workloads but is generally less
    reproducible. We keep deterministic mode as the opposite of benchmark so
    callers can choose a clear speed vs. reproducibility behavior.

    Args:
        benchmark: Whether to enable cuDNN benchmark mode.
    """
    torch.backends.cudnn.benchmark = bool(benchmark)
    torch.backends.cudnn.deterministic = not bool(benchmark)

def _get_device(gpu_id: int = 0) -> torch.device:
    """Resolve the runtime device from CUDA availability and requested GPU id.

    Args:
        gpu_id: CUDA device index. Values smaller than 0 force CPU execution.

    Returns:
        Selected `torch.device`.
    """
    if not torch.cuda.is_available() or gpu_id < 0:
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu_id}")

def initialize_logger(output_path: str, verbosity: int = 1):
    """Initialize the package logger.

    Args:
        output_path: Log file path for file handler output.
        verbosity: Verbosity level (0=warning, 1=info, 2=debug).

    Returns:
        Configured logger instance named `"dgs"`.

    Notes:
        If handlers already exist on the logger, this function returns early and
        does not add duplicate handlers.
    """
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
                      gpu: int = 0) -> Tuple[torch.device, logging.Logger]:
    """Set up runtime environment for DGS workflows.

    Args:
        output_dir: Directory used for run artifacts and logs.
        verbose: Verbosity level (0=warning, 1=info, 2=debug).
        seed: Random seed passed to Python, NumPy, and PyTorch.
        benchmark: Whether to enable cuDNN benchmark mode.
        gpu: CUDA device index. Values smaller than 0 force CPU.

    Returns:
        Tuple of `(device, logger)`.

    Side effects:
        - Creates `output_dir` if missing.
        - Creates and configures a log file in `output_dir`.
        - Sets Python/NumPy/PyTorch random seeds.
        - Optionally updates cuDNN benchmark configuration.
    """
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

    # Setup backend behavior
    _set_torch_backend(benchmark)
    if benchmark:
        logger.info("PyTorch benchmark mode enabled (deterministic=False)")
    else:
        logger.info("PyTorch benchmark mode disabled (deterministic=True)")

    # Setup device
    device = _get_device(gpu)
    logger.info(f"Using device: {device}")
        
    return device, logger

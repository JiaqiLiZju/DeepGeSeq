"""
Configuration Management Module for DGS (Deep Genomic Sequence Analysis Toolkit)

This module provides a comprehensive configuration system for managing all aspects
of genomic sequence analysis in DGS. It includes configuration classes for data
processing, model architecture, training, evaluation, and prediction.

Key Components:
1. Configuration Classes:
   - DataConfig: Data processing and dataset management
   - ModelConfig: Neural network architecture settings
   - TrainerConfig: Training parameters and optimization
   - EvaluateConfig: Model evaluation settings
   - ExplainConfig: Model interpretation parameters
   - PredictConfig: Variant effect prediction settings
   - DgsConfig: Main configuration container

2. Configuration Management:
   - ConfigManager: Handles loading, saving, and updating configurations
   - Example configuration generation
   - Configuration validation and error handling

Usage Example:
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config("config.json")

    # Generate example configuration
    config_manager.generate_example_config("minimal", "example_config.json")
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Literal, List
from dataclasses import dataclass, field, asdict

@dataclass
class DataConfig:
    """
    Configuration for genomic data processing and dataset management.

    This class defines settings for:
    - Genome and interval data sources
    - Target task specifications
    - Data splitting strategies
    - Dataset parameters

    Used by:
    - preprocess_data_for_train: Training data preparation
    - preprocess_data_for_evaluate: Evaluation data preparation
    - preprocess_data_for_explain: Model interpretation data
    - preprocess_data_for_predict: Prediction data preparation

    Attributes:
        genome_path (str): Path to reference genome FASTA file
        intervals_path (str): Path to genomic intervals BED file
        target_tasks (List[Dict]): List of target task specifications
        train_test_split (str): Data split strategy ('random_split' or 'chromosome_split')
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of data for validation
        test_chroms (List[str]): Chromosomes to use for testing
        val_chroms (List[str]): Chromosomes to use for validation
        batch_size (int): Batch size for data loading
        strand_aware (bool): Whether to consider DNA strand information
        sequence_length (int): Length of input sequences
    """
    # Dataset settings
    genome_path: str = ""  # Path to genome fasta file
    intervals_path: str = ""  # Path to intervals bed file
    
    # Target settings
    target_tasks: List[Dict[str, Any]] = field(default_factory=lambda: [{
        'task_name': 'default',
        'file_path': '',
        'file_type': 'bigwig',  # or 'bed'
        'bin_size': None,
        'aggfunc': 'mean'
    }])
    
   # Data splitting settings
    train_test_split: Literal["random_split", "chromosome_split"] = "random_split"
    test_size: float = 0.2
    val_size: float = 0.2
    test_chroms: List[str] = field(default_factory=lambda: ["chr8"])
    val_chroms: List[str] = field(default_factory=lambda: ["chr7"])
    
    # Dataset settings
    batch_size: int = 4
    strand_aware: bool = True
    sequence_length: int = 1000


@dataclass
class ModelConfig:
    """
    Configuration for neural network model architecture.

    This class defines the structure and parameters of the deep learning model
    used for genomic sequence analysis.

    Attributes:
        type (str): Model architecture type (e.g., 'CNN', 'Transformer')
        input_size (int): Length of input sequences
        output_size (int): Number of output features/tasks
        hidden_size (int): Size of hidden layers
        num_layers (int): Number of layers in the model
        dropout (float): Dropout rate for regularization
        activation (str): Activation function type
        normalization (str): Normalization layer type
    """
    type: str = "CNN"  # Model type (e.g., CNN, Transformer)
    input_size: int = 1000  # Input sequence length
    output_size: int = 1  # Number of output features
    hidden_size: int = 128  # Hidden layer size
    num_layers: int = 3  # Number of layers
    dropout: float = 0.1  # Dropout rate
    activation: str = "ReLU"  # Activation function
    normalization: str = "BatchNorm1d"  # Normalization layer

@dataclass
class TrainerConfig:
    """
    Configuration for model training and optimization.

    This class defines all parameters related to model training, including:
    - Optimizer settings and hyperparameters
    - Loss function configuration
    - Training control parameters
    - Checkpoint and logging settings

    Used by:
        execute_dgs_train: Main training execution function

    Attributes:
        optimizer (Dict): Optimizer type and parameters
        criterion (Dict): Loss function type and parameters
        patience (int): Early stopping patience
        max_epochs (int): Maximum training epochs
        clip_grad_norm (bool): Whether to clip gradients
        max_norm (float): Maximum gradient norm
        checkpoint_dir (str): Directory for saving checkpoints
        best_model_path (str): Path for saving best model
        resume (bool): Whether to resume training
        resume_model_name (str): Name of checkpoint to resume from
        use_tensorboard (bool): Whether to use TensorBoard
        tensorboard_dir (str): Directory for TensorBoard logs
    """

    # Optimizer settings
    optimizer: Dict[str, Any] = field(default_factory=lambda: {
        "type": "Adam",
        "params": {
            "lr": 1e-3,
            "weight_decay": 0
        }
    })

    # Loss function settings
    criterion: Dict[str, Any] = field(default_factory=lambda: {
        "type": "MSELoss",
        "params": {}
    })

    # early stopping settings
    patience: int = 10
    max_epochs: int = 500
    
    # Gradient settings
    clip_grad_norm: bool = False
    max_norm: float = 10.0
    
    # Output settings
    checkpoint_dir: str = "checkpoints"
    best_model_path: str = "checkpoints/best_model.pt"

    # resume settings
    resume: bool = False
    resume_model_name: str = "best_model.pt"

    # Tensorboard settings
    use_tensorboard: bool = True
    tensorboard_dir: str = "tensorboard"
    
    # # Evaluation settings
    # evaluate_training: bool = True
    # metric_sample: int = 100
    # item_sample: int = 5000
    
@dataclass
class EvaluateConfig:
    """
    Configuration for model evaluation and performance assessment.

    This class defines settings for evaluating trained models on test data
    and computing various performance metrics.

    Used by:
        execute_dgs_evaluate: Model evaluation function

    Attributes:
        split (str): Data split to evaluate on ('train', 'val', 'test')
        metrics (List[str]): List of metrics to compute
        output_dir (str): Directory for evaluation results
        save_predictions (bool): Whether to save model predictions
    """
    split: Literal["train", "val", "test"] = "test"
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "auc"
    ])
    output_dir: str = "evaluation_results"
    save_predictions: bool = True

@dataclass
class ExplainConfig:
    """
    Configuration for model interpretation and visualization.

    This class defines parameters for generating explanations of model
    predictions and identifying important sequence patterns.

    Attributes:
        target (int): Target task index for interpretation
        output_dir (str): Directory for saving interpretation results
        max_seqlets (int): Maximum number of sequence elements to analyze
    """
    target: int = 0
    output_dir: str = "motif_results"
    max_seqlets: int = 2000

@dataclass
class PredictConfig:
    """
    Configuration for variant effect prediction.

    This class defines settings for predicting the effects of genetic
    variants on sequence function.

    Attributes:
        vcf_path (str): Path to VCF file containing variants
        sequence_length (int): Length of sequence context around variants
        metric_func (str): Function for computing variant effects
        mean_by_tasks (bool): Whether to average predictions across tasks
    """
    vcf_path: str = ""
    sequence_length: int = 1000
    metric_func: str = "diff"
    mean_by_tasks: bool = True

@dataclass
class DgsConfig:
    """
    Main configuration class for DGS toolkit.

    This class serves as the central configuration container, combining
    all component-specific configurations and global settings.

    Attributes:
        modes (List[str]): List of analysis modes to run
        device (str): Computation device ('cuda' or 'cpu')
        output_dir (str): Main output directory
        data (DataConfig): Data processing configuration
        model (Dict): Model architecture configuration
        train (TrainerConfig): Training configuration
        explain (ExplainConfig): Interpretation configuration
        predict (PredictConfig): Prediction configuration
    """
    # Required configurations
    modes: List[Literal["train", "evaluate", "explain", "predict"]] = field(
        default_factory=lambda: ["train", "evaluate"]
    )
    device: str = "cuda"
    output_dir: str = "output"
    
    # Component configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: Dict[str, Any] = field(default_factory=lambda: {
        "type": "CNN",
        "args": {"output_size": 1}
    })
    train: TrainerConfig = field(default_factory=TrainerConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

class ConfigError(Exception):
    """
    Exception class for configuration-related errors.

    This class is used to raise errors when:
    - Required configuration parameters are missing
    - Parameter values are invalid
    - Configuration file cannot be loaded
    """

class ConfigManager:
    """
    Configuration management system for DGS.

    This class provides methods for:
    - Loading configurations from files or dictionaries
    - Saving configurations to files
    - Generating example configurations
    - Updating existing configurations
    - Validating configuration parameters
    """
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        
    def load_config(self, config: Union[str, Dict[str, Any], Path]) -> Dict[str, Any]:
        """
        Load configuration from a file or dictionary.

        Args:
            config: Configuration source (file path or dictionary)

        Returns:
            Dict[str, Any]: Loaded configuration dictionary

        Raises:
            FileNotFoundError: If config file not found
            ConfigError: If configuration is invalid
        """
        if isinstance(config, (str, Path)):
            config = self._load_from_file(config)
        
        self._config = config
        return config
    
    def _load_from_file(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path) as f:
            return json.load(f)
    
    def save_config(self, path: Union[str, Path]):
        """
        Save current configuration to a file.

        Args:
            path: Path to save configuration file
        """
        with open(path, 'w') as f:
            json.dump(self._config, f, indent=4)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update current configuration."""
        self._config.update(updates)

    def generate_example_config(self, example: Literal["minimal", "full"], output: str):
        """
        Generate an example configuration file.

        Args:
            example: Type of example configuration ('minimal' or 'full')
            output: Path to save the example configuration

        Raises:
            ValueError: If invalid example type specified
        """
        if example == "minimal":
            config = minimal_config
        elif example == "full":
            config = complete_configs
        else:
            raise ValueError(f"Invalid example: {example}")
        self._config = config
        self.save_config(output)

# minimal config
minimal_config = {
    "modes" : ["train", "evaluate", "explain", "predict"],
    "device": "cuda",
    "output_dir": "Test",
    "data": {
        "genome_path": "Test/reference_grch38p13/GRCh38.p13.genome.fa.gz",
        "intervals_path": "Test/random_regions.bed",
        "target_tasks": [
            {
                "task_name": "gc_content",
                "file_path": "Test/hg38.gc5Base.bw",
                "file_type": "bigwig",                
            },
            {
                "task_name": "recomb",
                "file_path": "Test/recombAvg.bw",
                "file_type": "bigwig",                
            }
        ]
    },
    "train": {
        "optimizer": {
            "type": "Adam",
            "lr": 0.001
        },
        "criterion": {
            "type": "MSELoss"
        }
    },
    "model": {
        "type": "CNN",
        "args":{"output_size": 2}
    },
    "explain":{"target":0},
    "predict":{"vcf_path":"Test/test.vcf",
              "sequence_length":1000
              }
}

# Example configurations for different scenarios
complete_configs = {
    "modes": ["train", "evaluate", "predict"],
    "device": "cuda",
    "output_dir": "Test",
    
    "data": {
        "genome_path": "Test/reference_grch38p13/GRCh38.p13.genome.fa.gz",
        "intervals_path": "Test/random_regions.bed",
        "target_tasks": [
            {
                "task_name": "gc_content",
                "file_path": "Test/hg38.gc5Base.bw",
                "file_type": "bigwig"
            },
            {
                "task_name": "recomb",
                "file_path": "Test/recombAvg.bw",
                "file_type": "bigwig"
            }
        ],
        "train_test_split": "random_split",
        "test_size": 0.2,
        "val_size": 0.2,
        "test_chroms": ["chr8"],
        "val_chroms": ["chr7"],
        "strand_aware": True,
        "batch_size": 4
    },
    
    "model": {
        "type": "CNN",
        "args": {"output_size": 2}
    },
    
    "train": {
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 1e-3}
        },
        "criterion": {
            "type": "MSELoss",
            "params": {}
        },
        "patience": 10,
        "max_epochs": 500,
        "checkpoint_dir": "checkpoints",
        "use_tensorboard": False,
        "tensorboard_dir": "tensorboard"
    },
    
    "explain": {
        "target": 0,
        "output_dir": "motif_results",
        "max_seqlets": 2000
    },
    
    "predict": {
        "vcf_path": "Test/test.vcf",
        "sequence_length": 1000,
        "metric_func": "diff",
        "mean_by_tasks": True
    }
}
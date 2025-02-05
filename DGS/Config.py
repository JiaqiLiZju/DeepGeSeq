"""Configuration management module for DGS.

This module provides configuration classes and management for DGS operations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Literal, List
from dataclasses import dataclass, field, asdict

@dataclass
class DataConfig:
    """Configuration for data processing.
    
    Used by:
    - preprocess_data_for_train
    - preprocess_data_for_evaluate
    - preprocess_data_for_explain
    - preprocess_data_for_predict
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
    """Configuration for model architecture."""
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
    """Configuration for model training.
    
    Used by execute_dgs_train
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
    """Configuration for model evaluation.
    
    Used by execute_dgs_evaluate
    """
    split: Literal["train", "val", "test"] = "test"
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "auc"
    ])
    output_dir: str = "evaluation_results"
    save_predictions: bool = True

@dataclass
class ExplainConfig:
    """Configuration for model interpretation."""
    target: int = 0
    output_dir: str = "motif_results"
    max_seqlets: int = 2000

@dataclass
class PredictConfig:
    """Configuration for variant effect prediction."""
    vcf_path: str = ""
    sequence_length: int = 1000
    metric_func: str = "diff"
    mean_by_tasks: bool = True

@dataclass
class DgsConfig:
    """Main configuration class for DGS."""
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
    """Configuration related errors."""
    pass

class ConfigManager:
    """Configuration manager for DGS."""
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        
    def load_config(self, config: Union[str, Dict[str, Any], Path]) -> Dict[str, Any]:
        """Load configuration from file or dictionary."""
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
        """Save current configuration to file."""
        with open(path, 'w') as f:
            json.dump(self._config, f, indent=4)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update current configuration."""
        self._config.update(updates)

    def generate_example_config(self, example: Literal["minimal", "full"], output: str):
        """Generate example configuration file."""
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
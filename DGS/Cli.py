"""
DGS Command Line Interface Implementation Module

This module implements the core functionality of the DGS (Deep Genomic Sequence Analysis Toolkit)
command line interface. It provides a comprehensive set of classes and functions for executing
various genomic analysis tasks.

Key Components:

1. DgsCLI Class:
   - Main client class for executing DGS commands
   - Handles configuration validation and component initialization
   - Manages execution flow for all analysis modes

2. Core Execution Functions:
   - execute_dgs_train: Model training with customizable parameters
   - execute_dgs_evaluate: Model evaluation and metrics calculation
   - execute_dgs_explain: Model interpretation and visualization
   - execute_dgs_predict: Sequence prediction and variant effect prediction
   - execute_dgs_hpo: Hyperparameter optimization (planned feature)

3. Data Preprocessing Functions:
   - preprocess_data_for_train: Prepare data for model training
   - preprocess_data_for_evaluate: Prepare data for model evaluation
   - preprocess_data_for_explain: Prepare data for model interpretation
   - preprocess_data_for_predict: Prepare data for sequence prediction

Each component is designed to work independently or as part of a complete analysis pipeline.
"""

from typing import Optional, Dict, Any

import os
import logging
import numpy as np

import torch
from torch import nn
from torch import optim

logger = logging.getLogger(__name__)

# TODO not implemented
# def execute_dgs_hpo(search_space=None):
#     from .Architecture import hyperparameter_tune
#     hyperparameter_tune(search_space=search_space, num_samples=10, max_num_epochs=10, gpus_per_trial=1)

def preprocess_data_for_train(genome_path, target_tasks, intervals_path, 
                              train_test_split="random_split",
                              test_size=0.2, val_size=0.2,
                              test_chroms=["chr8"], val_chroms=["chr7"],
                              strand_aware=True, batch_size=4):
    """
    Prepare genomic data for model training by loading and preprocessing sequences and targets.

    Args:
        genome_path (str): Path to the reference genome file
        target_tasks (list): List of target tasks to predict
        intervals_path (str): Path to the genomic intervals file
        train_test_split (str): Split strategy ('random_split' or 'chromosome_split')
        test_size (float): Proportion of data for testing (for random_split)
        val_size (float): Proportion of data for validation (for random_split)
        test_chroms (list): Chromosomes to use for testing (for chromosome_split)
        val_chroms (list): Chromosomes to use for validation (for chromosome_split)
        strand_aware (bool): Whether to consider DNA strand information
        batch_size (int): Batch size for data loading

    Returns:
        tuple: (train_loader, val_loader, test_loader) - DataLoader objects for each dataset
    """
    from .Data.Sequence import Genome
    from .Data.Target import Target
    from .Data.Interval import Interval
    from .Data.Dataset import GenomicDataset
    from .Data.Dataset import create_dataloader
    from .Data.Sampler import random_split, chromosome_split

    # load genome and intervals
    genome = Genome(genome_path)
    intervals = Interval(intervals_path)
    logger.info("Loaded genome and intervals")

    # train test split
    if train_test_split == "random_split":
        train_intervals, val_intervals, test_intervals = random_split(intervals, test_size=test_size, val_size=val_size)
    elif train_test_split == "chromosome_split":
        train_intervals, val_intervals, test_intervals = chromosome_split(intervals, test_chroms=test_chroms, val_chroms=val_chroms)
    logger.info("Split intervals")
    
    # load target
    train_target = Target(train_intervals.data, target_tasks)
    val_target = Target(val_intervals.data, target_tasks)
    test_target = Target(test_intervals.data, target_tasks)
    logger.info("Loaded target")

    # create dataset
    train_ds = GenomicDataset(train_intervals, genome, train_target, strand_aware=strand_aware)
    test_ds = GenomicDataset(test_intervals, genome, test_target, strand_aware=strand_aware)
    val_ds = GenomicDataset(val_intervals, genome, val_target, strand_aware=strand_aware)
    logger.info("Created dataset")

    # create dataloader
    train_loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = create_dataloader(test_ds, batch_size=batch_size, shuffle=False)
    val_loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    logger.info("Created dataloader")
    
    return train_loader, val_loader, test_loader

def execute_dgs_train(train_loader, validate_loader, 
                       model, optimizer, criterion, device, 
                       patience=10, max_epochs=500,
                       resume=False, resume_model_name="best_model.pt",
                       checkpoint_dir="checkpoints", 
                       use_tensorboard=True, tensorboard_dir="tensorboard", 
                       **trainer_args):
    """
    Execute model training with the specified parameters and data.

    Args:
        train_loader (DataLoader): DataLoader for training data
        validate_loader (DataLoader): DataLoader for validation data
        model (nn.Module): Neural network model to train
        optimizer (optim.Optimizer): Optimizer for model training
        criterion (callable): Loss function
        device (str): Device to use for training ('cuda' or 'cpu')
        patience (int): Number of epochs to wait for improvement before early stopping
        max_epochs (int): Maximum number of training epochs
        resume (bool): Whether to resume training from a checkpoint
        resume_model_name (str): Name of the checkpoint file to resume from
        checkpoint_dir (str): Directory for saving checkpoints
        use_tensorboard (bool): Whether to use TensorBoard for logging
        tensorboard_dir (str): Directory for TensorBoard logs
        **trainer_args: Additional arguments for the Trainer

    Returns:
        tuple: (model, trainer) - Trained model and trainer instance
    """
    from .DL.Trainer import Trainer

    # initialize trainer
    trainer = Trainer(model, criterion, optimizer, device, 
                      checkpoint_dir=checkpoint_dir, 
                      use_tensorboard=use_tensorboard, tensorboard_dir=tensorboard_dir,
                      patience=patience, **trainer_args)
    logger.info("Initialized trainer")

    # resume from checkpoint
    if resume:
        # check if checkpoint exists
        if not os.path.exists(os.path.join(checkpoint_dir, resume_model_name)):
            raise FileNotFoundError(f"Checkpoint file {resume_model_name} not found in {checkpoint_dir}")
        trainer.load_checkpoint(os.path.join(checkpoint_dir, resume_model_name))
        logger.info("Resumed from checkpoint")
    
    # train
    trainer.train(train_loader, validate_loader, epochs=max_epochs, early_stopping=True)
    logger.info("Training completed")

    # reload best model
    trainer.load_checkpoint(os.path.join(checkpoint_dir, "best_model.pt"))
    logger.info("Reloaded best model")

    return trainer.model, trainer


def preprocess_data_for_evaluate(genome_path, target_task, intervals_path,
                                  train_test_split="random_split", test_size=0.2, test_chroms=["chr8"],
                                  strand_aware=True, batch_size=4):
    from .Data.Sequence import Genome
    from .Data.Target import Target
    from .Data.Interval import Interval
    from .Data.Dataset import GenomicDataset
    from .Data.Dataset import create_dataloader
    from .Data.Sampler import chromosome_split, random_split

    # load genome and intervals
    genome = Genome(genome_path)
    intervals = Interval(intervals_path)
    logger.info("Loaded genome and intervals")

    # train test split
    if train_test_split == "chromosome_split":
        _, test_intervals = chromosome_split(intervals, test_chroms=test_chroms)
    elif train_test_split == "random_split":
        _, test_intervals = random_split(intervals, test_size=test_size)
    logger.info("Split intervals")

    # load target
    target = Target(test_intervals.data, target_task)
    logger.info("Loaded target")

    # create dataset
    test_ds = GenomicDataset(test_intervals, genome, target, strand_aware=strand_aware)
    logger.info("Created dataset")

    # create dataloader
    test_loader = create_dataloader(test_ds, batch_size=batch_size, shuffle=False)
    logger.info("Created dataloader")

    return test_loader

def execute_dgs_evaluate(test_loader, trainer):
    """
    Evaluate a trained model and calculate performance metrics.

    Args:
        test_loader (DataLoader): DataLoader containing test data
        trainer (Trainer): Trained model trainer instance

    Returns:
        dict: Dictionary containing evaluation metrics:
            - For classification: accuracy, precision, recall, F1-score, ROC-AUC
            - For regression: MSE, MAE, R2 score, correlation
            - For sequence tasks: position-wise metrics
    """
    from .DL.Evaluator import calculate_classification_metrics, calculate_regression_metrics
    from .DL.Evaluator import calculate_sequence_classification_metrics, calculate_sequence_regression_metrics

    # reload best model
    trainer.load_checkpoint(os.path.join(trainer.checkpoint_dir, "best_model.pt"))
    logger.info("Reloaded best model")

    # evaluate
    avg_loss, metric, predictions, targets = trainer.validate(test_loader, return_predictions=True)
    logger.info("Evaluated model")

    # calculate metrics
    if np.all(np.isin(targets, [0, 1])):
        target_task = 'classification'
    else:
        target_task = 'regression'

    if target_task == 'classification':
        if predictions.ndim == 3:
            metrics = calculate_sequence_classification_metrics(predictions, targets)
        else:
            metrics = calculate_classification_metrics(predictions, targets)
    elif target_task == 'regression':
        if predictions.ndim == 3:
            metrics = calculate_sequence_regression_metrics(predictions, targets)
        else:
            metrics = calculate_regression_metrics(predictions, targets)
    logger.info("Calculated metrics")

    return metrics


def preprocess_data_for_explain(genome_path, intervals_path,
                                train_test_split="random_split", test_size=0.2, test_chroms=["chr8"],
                                strand_aware=True, batch_size=4):
    """
    Prepare genomic data for model interpretation and explanation.

    Args:
        genome_path (str): Path to the reference genome file
        intervals_path (str): Path to the genomic intervals file
        train_test_split (str): Split strategy ('random_split' or 'chromosome_split')
        test_size (float): Proportion of data for testing (for random_split)
        test_chroms (list): Chromosomes to use for testing (for chromosome_split)
        strand_aware (bool): Whether to consider DNA strand information
        batch_size (int): Batch size for data loading

    Returns:
        DataLoader: DataLoader containing sequences for interpretation
    """
    from .Data.Sequence import Genome
    from .Data.Interval import Interval
    from .Data.Dataset import SeqDataset
    from .Data.Dataset import create_dataloader
    from .Data.Sampler import chromosome_split, random_split

    # load genome and intervals
    genome = Genome(genome_path)
    intervals = Interval(intervals_path)
    logger.info("Loaded genome and intervals")

    # split intervals
    if train_test_split == "chromosome_split":
        _, test_intervals = chromosome_split(intervals, test_chroms=test_chroms)
    elif train_test_split == "random_split":
        _, test_intervals = random_split(intervals, test_size=test_size)
    logger.info("Split intervals")

    # create dataset
    test_ds = SeqDataset(test_intervals, genome, strand_aware=strand_aware)
    logger.info("Created dataset")

    # create dataloader
    test_loader = create_dataloader(test_ds, batch_size=batch_size, shuffle=False)
    logger.info("Created dataloader")

    return test_loader

def execute_dgs_model_predict(model, test_loader, target, device, output_dir="motif_results", max_seqlets=2000):
    """
    Execute model prediction on test sequences.

    Args:
        model (nn.Module): Trained neural network model
        test_loader (DataLoader): DataLoader containing test sequences
        target (str): Target task for prediction
        device (str): Device to use for prediction ('cuda' or 'cpu')
        output_dir (str): Directory to save prediction results
        max_seqlets (int): Maximum number of sequence elements to process

    Note:
        Results will be saved to the specified output directory
    """
    from .DL.Predict import predict
    predict(model, test_loader, target=target, device=device,
            output_dir=output_dir, max_seqlets=max_seqlets)
    logger.info("Model prediction completed")

def execute_dgs_explain(model, test_loader, target, device, output_dir="motif_results", max_seqlets=2000):
    """
    Generate model explanations and motif enrichment analysis.

    Args:
        model (nn.Module): Trained neural network model
        test_loader (DataLoader): DataLoader containing test sequences
        target (str): Target task for explanation
        device (str): Device to use for computation ('cuda' or 'cpu')
        output_dir (str): Directory to save explanation results
        max_seqlets (int): Maximum number of sequence elements to analyze

    Note:
        Motif enrichment results will be saved to the specified output directory
    """
    from .DL.Explain import motif_enrich
    motif_enrich(model, test_loader.dataset, target=target, device=device,
                 output_dir=output_dir, max_seqlets=max_seqlets)
    logger.info("Motif enrichment completed")

def preprocess_data_for_predict(genome_path, vcf_filename, target_len=1000):
    """
    Prepare genomic data for variant effect prediction.

    Args:
        genome_path (str): Path to the reference genome file
        vcf_filename (str): Path to the VCF file containing variants
        target_len (int): Length of sequence context around variants

    Returns:
        tuple: (VariantDataset, DataFrame) - Dataset for variant prediction and variant information
    """
    from .Data.Sequence import Genome
    from .DL.Predict import read_vcf
    from .DL.Predict import VariantDataset
    
    # load genome
    genome = Genome(genome_path)
    logger.info("Loaded genome")
    
    # load vcf
    variant_df = read_vcf(vcf_filename)
    logger.info("Loaded vcf")
    
    # create vds
    vds = VariantDataset(genome, variant_df, target_len=target_len)
    logger.info("Created vds")

    return vds, variant_df

def execute_dgs_predict(model, vds, variant_df, metric_func, mean_by_tasks, device):
    """
    Execute variant effect prediction.

    Args:
        model (nn.Module): Trained neural network model
        vds (VariantDataset): Dataset containing variant sequences
        variant_df (DataFrame): DataFrame containing variant information
        metric_func (callable): Function to calculate prediction metrics
        mean_by_tasks (bool): Whether to average predictions across tasks
        device (str): Device to use for prediction ('cuda' or 'cpu')

    Returns:
        DataFrame: Prediction results for each variant
    """
    from .DL.Predict import vep_centred_on_ds
    
    # calculate P_diff
    P_diff = vep_centred_on_ds(model, vds, 
                               metric_func=metric_func, 
                               mean_by_tasks=mean_by_tasks, 
                               device=device)
    logger.info("Calculated P_diff")

    # add P_diff to variant_df
    variant_df['P_diff'] = P_diff
    logger.info("Added P_diff to variant_df")

    return variant_df


# client class
class DgsCLI:
    """
    Main client class for executing DGS commands.

    This class handles:
    - Configuration validation and management
    - Model and component initialization
    - Execution of different analysis modes (train, evaluate, explain, predict)
    - Resource management and cleanup

    Attributes:
        config (dict): Configuration dictionary containing all parameters
        device (str): Device to use for computation ('cuda' or 'cpu')
        model (nn.Module): Neural network model instance
        optimizer (optim.Optimizer): Optimizer for model training
        criterion (callable): Loss function for training
    """
    
    def __init__(self, config, device=None):
        """
        Initialize DgsCLI instance.

        Args:
            config (dict): Configuration dictionary containing all parameters
            device (str, optional): Device to use for computation. Defaults to None.
        """
        self.logger = logging.getLogger("dgs")

        # Validate essential config keys
        self.config = config
        self._validate_minimal_config()

        if device is None and 'device' in self.config:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize components
        self._initialize_components()
    
    def _validate_minimal_config(self):
        """
        Validate the minimal required configuration parameters.

        Raises:
            ConfigError: If required parameters are missing or invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")
        
        if "data" not in self.config:
            raise ValueError("Config must contain 'data' key")
        
        if "model" not in self.config:
            self.config["model"] = {"type": "CNN"}
            self.config["model"]["args"] = {"output_size": len(self.config["data"]["target_tasks"])}
            self.logger.warning("No model specified in config, using default model: %s", self.config["model"])
        
        if "modes" not in self.config:
            self.config["modes"] = ["train", "evaluate"]
            self.logger.warning("No modes specified in config, using default modes: %s, \n \
                                if you want to use explain mode, please specify the target in config, \n \
                                if you want to use prediction mode, please specify the VCF path in config", 
                                self.config["modes"])

        if "device" not in self.config:
            self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.warning("No device specified in config, using default device: %s", self.config["device"])

        if "output_dir" not in self.config:
            self.config["output_dir"] = os.path.join(os.getcwd(), "output")
            self.logger.warning("No output_dir specified in config, using default output_dir: %s", self.config["output_dir"])

    def _initialize_components(self):
        """
        Initialize all required components based on configuration.

        This includes:
        - Model architecture
        - Optimizer
        - Loss function
        - Data preprocessing components
        """
        self.logger.info("Initializing components...")

        # Initialize model
        self.logger.info("Initializing model...")
        try:
            # from .Config import get_model_from_config
            # self.model = get_model_from_config(self.config["model"])
            from . import Model
            model_config = self.config["model"]
            model_type = model_config["type"]
            model = getattr(Model, model_type)
            self.model = model(**model_config.get("args", {}))
            self.model = self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}")
            
        # Initialize train components
        if "train" in self.config:
            self.logger.info("Initializing optimizer...")
            try:
                from torch import optim
                optimizer_config = self.config["train"]["optimizer"]
                optimizer_class = getattr(optim, optimizer_config["type"])
                self.optimizer = optimizer_class(
                    self.model.parameters(),
                    **optimizer_config.get("params", {})
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize optimizer: {e}")
            
            self.logger.info("Initializing criterion...")
            try:
                from torch import nn
                criterion_config = self.config["train"]["criterion"]
                criterion_class = getattr(nn, criterion_config["type"])
                self.criterion = criterion_class(**criterion_config.get("params", {}))
            except Exception as e:
                raise RuntimeError(f"Failed to initialize criterion: {e}")

        # initialize trainer
        if self.optimizer is not None and self.criterion is not None:
            self.logger.info("Initializing trainer...")
            try:
                from .DL.Trainer import Trainer
                self.trainer = Trainer(
                    self.model, 
                    self.criterion, 
                    self.optimizer, 
                    self.device, 
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize trainer: {e}")
    
    def execute(self):
        """
        Execute the requested command based on configuration.

        This method dispatches to the appropriate execution method based on
        the command specified in the configuration (train, evaluate, explain, predict).

        Raises:
            ValueError: If an invalid command is specified
        """
    
        # Training mode
        if "train" in self.config.get("modes", []):
            self.logger.info("Starting training mode...")
            self._execute_train()
        
        # Evaluation mode
        if "evaluate" in self.config.get("modes", []):
            self.logger.info("Starting evaluation mode...")
            self._execute_evaluate()
        
        # Explanation mode
        if "explain" in self.config.get("modes", []):
            self.logger.info("Starting explanation mode...")
            self._execute_explain()
            
        # Prediction mode
        if "predict" in self.config.get("modes", []):
            self.logger.info("Starting prediction mode...")
            self._execute_predict()
    
    def _execute_train(self):
        """Execute training mode."""
        # Prepare data
        self.logger.info("Preparing data for training...")
        train_loader, val_loader, test_loader = preprocess_data_for_train(
            self.config["data"]["genome_path"],
            self.config["data"]["target_tasks"],
            self.config["data"]["intervals_path"],
            train_test_split=self.config["data"].get("train_test_split", "random_split"),
            test_size=self.config["data"].get("test_size", 0.2),
            val_size=self.config["data"].get("val_size", 0.2),
            test_chroms=self.config["data"].get("test_chroms", ["chr8"]),
            val_chroms=self.config["data"].get("val_chroms", ["chr7"]),
            strand_aware=self.config["data"].get("strand_aware", True),
            batch_size=self.config["data"].get("batch_size", 4)
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Train model
        self.logger.info("Training model...")
        self.model, self.trainer = execute_dgs_train(
            self.train_loader,
            self.val_loader,
            self.model,
            self.optimizer,
            self.criterion,
            self.device,
            patience=self.config["train"].get("patience", 10),
            max_epochs=self.config["train"].get("max_epochs", 500),
            resume=self.config["train"].get("resume", False),
            resume_model_name=self.config["train"].get("resume_model_name", "best_model.pt"),
            checkpoint_dir=self.config["train"].get("checkpoint_dir", "checkpoints"), 
            use_tensorboard=self.config["train"].get("use_tensorboard", False),
            tensorboard_dir=self.config["train"].get("tensorboard_dir", "tensorboard"), 
            **self.config["train"].get("trainer_args", {})
        )
        self.logger.info("Training completed.")

    def _execute_evaluate(self):
        """Execute evaluation mode."""        
        if not hasattr(self, "trainer"):
            raise RuntimeError("No trained model available for evaluation, \n \
                               please train the model first, or specify the model in config")
        
        self.logger.info("Preparing data for evaluation...")
        if not hasattr(self, "test_loader"):
            self.test_loader = preprocess_data_for_evaluate(
                self.config["data"]["genome_path"],
                self.config["data"]["target_tasks"],
                self.config["data"]["intervals_path"],
                train_test_split=self.config["data"].get("train_test_split", "random_split"),
                test_size=self.config["data"].get("test_size", 0.2),
                test_chroms=self.config["data"].get("test_chroms", ["chr8"]),
                strand_aware=self.config["data"].get("strand_aware", True),
                batch_size=self.config["data"].get("batch_size", 4)
            )
        
        self.logger.info("Evaluating model...")
        self.metrics = execute_dgs_evaluate(
            self.test_loader, 
            self.trainer
        )

        # save metrics
        self.metrics.to_csv(os.path.join(self.config["output_dir"], "metrics.csv"))
        self.logger.info("Evaluation completed.")
    
    def _execute_explain(self):
        """Execute explanation mode."""
        if not hasattr(self, "model"):
            raise RuntimeError("No model available for explanation")
        
        self.logger.info("Preparing data for explanation...")
        if not hasattr(self, "explain_data_loader"):
            self.explain_data_loader = preprocess_data_for_explain(
                self.config["data"]["genome_path"],
                self.config["data"]["intervals_path"],
                train_test_split=self.config["data"].get("train_test_split", "random_split"),
                test_size=self.config["data"].get("test_size", 0.2),
                test_chroms=self.config["data"].get("test_chroms", ["chr8"]),
                strand_aware=self.config["data"].get("strand_aware", True),
                batch_size=self.config["data"].get("batch_size", 4)
            )
        
        self.logger.info("Executing explanation...")
        execute_dgs_explain(
            self.model,
            self.explain_data_loader,
            target=self.config["explain"].get("target", 0),
            device=self.device,
            output_dir=self.config["explain"].get("motif_results", "motif_results"), 
            max_seqlets=self.config["explain"].get("max_seqlets", 2000)
        )
        self.logger.info("Explanation completed.")
    
    def _execute_predict(self):
        """Execute prediction mode."""
        if not hasattr(self, "model"):
            raise RuntimeError("No model available for prediction")

        self.logger.info("Preparing data for prediction...")
        if not hasattr(self, "predict_data_loader"):
            self.vds, self.variant_df  = preprocess_data_for_predict(
                self.config["data"]["genome_path"],
                self.config["predict"]["vcf_path"],
                target_len=self.config["predict"]["sequence_length"]
            )
        
        self.logger.info("Predicting variants...")
        self.result_df = execute_dgs_predict(
            self.model,
            self.vds,
            self.variant_df,
            metric_func=self.config["predict"].get("metric_func", "diff"),
            mean_by_tasks=self.config["predict"].get("mean_by_tasks", True),
            device=self.device
        )

        save_path = os.path.join(self.config["output_dir"], "variant_predictions.csv")
        self.result_df.to_csv(save_path)
        self.logger.info("Variant predictions saved to %s", save_path)
        self.logger.info("Prediction completed.")

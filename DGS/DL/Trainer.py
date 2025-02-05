"""Trainer module for model training and evaluation in DGS.

This module provides:
1. Trainer class for model training and evaluation
2. Training utilities and callbacks
3. Metrics tracking and logging
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

logger = logging.getLogger("dgs.trainer")

@dataclass
class TrainerMetrics:
    """Training metrics tracker."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_metrics: List[float] = field(default_factory=list)
    val_metrics: List[float] = field(default_factory=list)
    best_val_loss: float = float('inf')
    best_val_metric: float = 0.0
    best_epoch: int = 0
    
@dataclass
class TrainerState:
    """Trainer state for checkpointing."""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    model_state: Dict = field(default_factory=dict)
    optimizer_state: Dict = field(default_factory=dict)
    metrics: TrainerMetrics = field(default_factory=TrainerMetrics)

class Trainer:
    """Model trainer for deep learning models.
    
    Handles:
    - Model training and validation
    - Metrics tracking and early stopping
    - Checkpointing and model saving
    - Prediction and evaluation
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        scheduler: Optional[Any] = None,
        clip_grad_norm: bool = False,
        max_grad_norm: float = 1.0,
        evaluate_training: bool = False,
        metric_sample: int = 100,
        patience: int = 10,
        use_tensorboard: bool = False,
        tensorboard_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Neural network model
            criterion: Loss criterion
            optimizer: Optimizer
            device: Computation device
            checkpoint_dir: Directory for saving checkpoints
            scheduler: Learning rate scheduler
            clip_grad_norm: Whether to clip gradients
            max_grad_norm: Maximum gradient norm
            evaluate_training: Whether to evaluate during training
            metric_sample: Number of samples for metric calculation
            patience: Early stopping patience
            use_tensorboard: Whether to use tensorboard
            tensorboard_dir: Tensorboard log directory
        """
        self.device = device
        
        # Move model and criterion to device
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        
        # Initialize optimizer
        self.optimizer = optimizer
        
        # Ensure optimizer's parameters are on the correct device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        self.scheduler = scheduler
        
        # Training settings
        self.clip_grad_norm = clip_grad_norm
        self.max_grad_norm = max_grad_norm
        self.evaluate_training = evaluate_training
        self.metric_sample = metric_sample
        self.patience = patience
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir or "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard = SummaryWriter(tensorboard_dir or "runs")
            
        # Initialize state
        self.state = TrainerState()
        self.metrics = TrainerMetrics()
        
    def _prepare_batch(self, data, target):
        """Prepare batch data by ensuring all inputs are tensors on the correct device.
        
        Args:
            data: Input data, can be:
                - single tensor/array/list
                - list of tensors/arrays/lists
            target: Target data, same format options as data
                
        Returns:
            (tensor or list[tensor], tensor or list[tensor]): 
                Processed data and target tensors on the correct device
        """
        def _to_tensor(x, dtype=None):
            if isinstance(x, torch.Tensor):
                x = x.to(self.device)
                return x if dtype is None else x.to(dtype)
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).to(self.device)
                return x if dtype is None else x.to(dtype)
            x = torch.tensor(x, device=self.device)
            return x if dtype is None else x.to(dtype)
        
        def _process_input(x, dtype=None):
            if isinstance(x, (list, tuple)):
                return [_to_tensor(item, dtype) for item in x]
            return _to_tensor(x, dtype)
        
        # Process data with float32 dtype for model input
        processed_data = _process_input(data, dtype=torch.float32)
        
        # Process target with float32 dtype by default
        # Only try to get criterion dtype if it has parameters and is not None
        target_dtype = torch.float32
        if self.criterion is not None and hasattr(self.criterion, 'parameters'):
            try:
                param = next(self.criterion.parameters())
                if param is not None:
                    target_dtype = param.dtype
            except StopIteration:
                pass
                
        processed_target = _process_input(target, dtype=target_dtype)
            
        return processed_data, processed_target
        
    def save_checkpoint(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save training checkpoint.
        
        Args:
            path: Path to save checkpoint. If None, uses default path.
        """
        path = Path(path or self.checkpoint_dir / f"checkpoint_{self.state.epoch}.pt")
        
        # Move model to CPU before saving
        model_state = {
            k: v.cpu() for k, v in self.model.state_dict().items()
        }
        
        # Move optimizer state to CPU
        optimizer_state = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in self.optimizer.state_dict().items()
        }
        
        # Update state
        self.state.model_state = model_state
        self.state.optimizer_state = optimizer_state
        self.state.metrics = self.metrics
        
        # Save checkpoint
        torch.save(self.state, path)
        logger.info(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Raises:
            FileNotFoundError: If checkpoint file not found
            RuntimeError: If checkpoint loading fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
            
        try:
            # Load checkpoint
            checkpoint = torch.load(path, map_location=self.device)
            
            # Restore state
            self.state = checkpoint
            self.metrics = checkpoint.metrics
            
            # Restore model and optimizer
            self.model.load_state_dict(checkpoint.model_state)
            self.optimizer.load_state_dict(checkpoint.optimizer_state)
            
            # Ensure optimizer state is on correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            
            logger.info(f"Loaded checkpoint from {path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
            
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        validate_fn: Optional[Callable] = None
    ) -> float:
        """Train one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            validate_fn: Optional validation function
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                try:
                    # Prepare and convert batch data
                    data, target = self._prepare_batch(data, target)
                    
                    # Forward pass (handle list of inputs if needed)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    
                    # Handle loss computation with multiple targets if needed
                    loss = self.criterion(output, target)
                    
                    # Backward pass
                    loss.backward()
                    if self.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )
                    self.optimizer.step()
                    
                    # Update metrics
                    total_loss += loss.item()
                    avg_loss = total_loss / (batch_idx + 1)
                    
                    # Update progress bar
                    pbar.set_postfix(loss=f"{avg_loss:.4f}")
                    
                    # Log to tensorboard
                    if self.use_tensorboard:
                        self.tensorboard.add_scalar(
                            "train/loss",
                            loss.item(),
                            self.state.global_step
                        )
                        
                    self.state.global_step += 1
                    
                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {e}")
                    continue
                    
        avg_loss = total_loss / len(train_loader)
        self.metrics.train_losses.append(avg_loss)
        
        return avg_loss
        
    def validate(
        self,
        val_loader: DataLoader,
        return_predictions: bool = False
    ) -> Union[Tuple[float, float], Tuple[float, float, torch.Tensor, torch.Tensor]]:
        """Validate model.
        
        Args:
            val_loader: Validation data loader
            return_predictions: Whether to return predictions
            
        Returns:
            Tuple of (validation loss, validation metric)
            If return_predictions=True, also returns (predictions, targets)
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                try:
                    # Prepare and convert batch data
                    data, target = self._prepare_batch(data, target)
                    
                    # Forward pass
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    # Store results
                    total_loss += loss.item()
                    if return_predictions:
                        predictions.append(output.cpu())
                        targets.append(target.cpu())
                        
                except Exception as e:
                    logger.error(f"Error in validation: {e}")
                    continue
                    
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        self.metrics.val_losses.append(avg_loss)
        
        if return_predictions:
            predictions = torch.cat(predictions)
            targets = torch.cat(targets)
            metric = self.compute_metric(predictions, targets)
            return avg_loss, metric, predictions, targets
            
        return avg_loss, 0.0
            
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping: bool = True,
        verbose: bool = True
    ) -> TrainerMetrics:
        """Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping: Whether to use early stopping
            verbose: Whether to show training logs
            
        Returns:
            Training metrics
        """
        start_time = time.time()
        no_improve = 0
        
        try:
            for epoch in range(self.state.epoch, epochs):
                epoch_start_time = time.time()
                self.state.epoch = epoch
                
                # Training
                train_loss = self.train_epoch(train_loader, epoch)
                
                # Validation
                val_loss, val_metric = self.validate(val_loader)
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                        current_lr = self.optimizer.param_groups[0]['lr']
                    else:
                        self.scheduler.step()
                        current_lr = self.scheduler.get_last_lr()[0]
                    
                # Save best model
                if val_loss < self.metrics.best_val_loss:
                    self.metrics.best_val_loss = val_loss
                    self.metrics.best_val_metric = val_metric
                    self.metrics.best_epoch = epoch
                    self.save_checkpoint(self.checkpoint_dir / "best_model.pt")
                    no_improve = 0
                else:
                    no_improve += 1
                    
                # Show training log
                epoch_time = time.time() - epoch_start_time
                current_metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_metric': val_metric
                }
                
                if self.scheduler is not None:
                    current_metrics['learning_rate'] = current_lr
                
                show_train_log(
                    epoch=epoch,
                    metrics=self.metrics,
                    current_metrics=current_metrics,
                    time_used=epoch_time,
                    save_dir=self.checkpoint_dir,
                    verbose=verbose,
                    plot=True
                )
                
                # Early stopping
                if early_stopping and no_improve >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
            
        finally:
            # Save final checkpoint
            self.save_checkpoint(self.checkpoint_dir / "final_model.pt")
            
            # Training summary
            duration = time.time() - start_time
            logger.info(
                f"\nTraining completed in {duration:.2f}s\n"
                f"Best validation loss: {self.metrics.best_val_loss:.4f}\n"
                f"Best validation metric: {self.metrics.best_val_metric:.4f}\n"
                f"Best epoch: {self.metrics.best_epoch}"
            )
            
        return self.metrics
        
    def predict(
        self,
        test_loader: DataLoader,
        return_probs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Make predictions.
        
        Args:
            test_loader: Test data loader
            return_probs: Whether to return probabilities
            
        Returns:
            Model predictions
            If return_probs=True, returns (predictions, probabilities)
        """
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for data in test_loader:
                try:
                    # Handle both (data) and (data, target) formats
                    if isinstance(data, (tuple, list)):
                        data = data[0]
                    
                    # Move data to device
                    data = data.to(self.device)
                    
                    # Forward pass
                    output = self.model(data)
                    
                    # Store results
                    predictions.append(output.cpu())
                    if return_probs:
                        probabilities.append(torch.sigmoid(output).cpu())
                        
                except Exception as e:
                    logger.error(f"Error in prediction: {e}")
                    continue
                    
        predictions = torch.cat(predictions)
        
        if return_probs:
            probabilities = torch.cat(probabilities)
            return predictions, probabilities
        return predictions
        
    @staticmethod
    def compute_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute evaluation metric.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Metric value
        """
        # Default to binary accuracy
        predictions = (torch.sigmoid(predictions) > 0.5).float()
        targets = (torch.sigmoid(targets) > 0.5).float()
        return (predictions == targets).float().mean().item()


def show_train_log(
    epoch: int,
    metrics: TrainerMetrics,
    current_metrics: Dict[str, float],
    time_used: float,
    verbose: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
    plot: bool = True
) -> None:
    """Display and visualize training log information.
    
    Args:
        epoch: Current epoch number
        metrics: TrainerMetrics instance containing training history
        current_metrics: Current epoch metrics dictionary containing:
            - train_loss: Training loss
            - val_loss: Validation loss
            - val_metric: Validation metric
            - learning_rate: Optional current learning rate
        time_used: Time used in seconds
        save_dir: Directory to save plots (optional)
        verbose: Whether to print log
        plot: Whether to plot training curves
    """
    # Text logging
    log_str = (
        f"\n{'='*50}\n"
        f"Epoch: {epoch}\n"
        f"Time: {time_used:.2f}s\n"
        f"Training Loss: {current_metrics['train_loss']:.4f}\n"
        f"Validation Loss: {current_metrics['val_loss']:.4f}\n"
        f"Validation Metric: {current_metrics['val_metric']:.4f}\n"
        f"Best Validation Loss: {metrics.best_val_loss:.4f}\n"
        f"Best Validation Metric: {metrics.best_val_metric:.4f}\n"
        f"Best Epoch: {metrics.best_epoch}\n"
    )
    
    if 'learning_rate' in current_metrics:
        log_str += f"Learning Rate: {current_metrics['learning_rate']:.6f}\n"
        
    log_str += f"{'='*50}\n"
    
    # Log to logger and print
    if verbose:
        logger.info(log_str)

    # Plot training curves
    if plot and save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot losses
        plot_curves(
            y_values=[metrics.train_losses, metrics.val_losses],
            labels=['Training', 'Validation'],
            title='Loss Curves',
            ylabel='Loss',
            save_path=save_dir / 'loss_curves.png'
        )
        
        # Plot metrics
        if metrics.val_metrics:
            plot_curves(
                y_values=[metrics.val_metrics],
                labels=['Validation'],
                title='Metric Curves',
                ylabel='Metric',
                save_path=save_dir / 'metric_curves.png'
            )

def plot_curves(
    y_values: List[List[float]],
    labels: List[str],
    title: str,
    ylabel: str,
    save_path: Union[str, Path],
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """Plot training curves.
    
    Args:
        y_values: List of y-values to plot
        labels: Labels for each curve
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save plot
        figsize: Figure size (width, height)
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    epochs = range(1, len(y_values[0]) + 1)
    
    for y, label in zip(y_values, labels):
        plt.plot(epochs, y, label=label)
        
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()

def plot_learning_rate(
    lr_history: List[float],
    save_path: Union[str, Path]
) -> None:
    """Plot learning rate curve.
    
    Args:
        lr_history: List of learning rates
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(lr_history) + 1)
    
    plt.plot(epochs, lr_history)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()


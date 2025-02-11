"""Basic modules for deep genomic sequence analysis.

This module provides fundamental building blocks and base classes for the DGS framework.
It includes abstract base classes, basic neural network modules, and utility classes
that form the foundation for more complex model architectures.

Classes
-------
BasicModule
    Abstract base class with weight initialization and model I/O utilities.
    Provides methods for parameter initialization and model saving/loading.

BasicConv1d
    Basic 1D convolution module with batch normalization, activation, and pooling.
    Designed for processing genomic sequence data.

BasicRNNModule
    Basic RNN/LSTM module configured for batch-first processing.
    Suitable for sequential genomic data analysis.

BasicLinearModule
    Linear transformation module with batch normalization and activation.
    Used for feature transformation and dimensionality reduction.

BasicPredictor
    Task-specific prediction head supporting various prediction types.
    Handles binary classification, multi-class classification, and regression.

BasicLoss
    Task-specific loss functions module.
    Provides appropriate loss functions for different prediction tasks.

Notes
-----
All modules follow a consistent interface and support debug-level logging for
shape tracking and parameter initialization monitoring.
"""

import random, time, logging
import numpy as np

import torch
from torch import nn

__all__ = ["BasicModule", "BasicConv1d", "BasicRNNModule", "BasicLinearModule", "BasicPredictor", "BasicLoss", "Flatten"]


class BasicModule(nn.Module):
    """Abstract base class for all modules in DGS.

    Provides common functionality for weight initialization, model saving/loading,
    and parameter management. All model classes should inherit from this class
    to ensure consistent behavior.

    Methods
    -------
    initialize_weights()
        Initialize module parameters using appropriate initialization schemes
    initialize_weights_from_pretrained(pretrained_net_fname)
        Load and initialize weights from a pretrained model
    load(path)
        Load model weights from a saved file
    save(fname=None)
        Save model weights to a file
    test(input_size)
        Test the module by running a forward pass with dummy input
    """
    def __init__(self):
        super(BasicModule,self).__init__()
        logging.debug("Initializing BasicModule")

    def initialize_weights(self):
        """Initialize module parameters using appropriate schemes.

        Applies different initialization methods based on layer type:
        - Conv layers: Xavier normal for weights, zero for bias
        - Linear layers: Xavier normal for weights, zero for bias
        - BatchNorm layers: Constant 1 for weights, 0 for bias
        - LSTM layers: Orthogonal initialization for all weights
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_normal_(m.weight) 
                if m.bias is not None:
                    m.bias.data.zero_()
                logging.debug("Initialized Conv layer: %s", str(m))

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight) 
                if m.bias is not None:
                    m.bias.data.zero_()
                logging.debug("Initialized Linear layer: %s", str(m))

            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                logging.debug("Initialized BatchNorm layer: %s", str(m))

            elif isinstance(m, nn.LSTM):
                nn.init.orthogonal_(m.all_weights[0][0])
                nn.init.orthogonal_(m.all_weights[0][1])
                nn.init.orthogonal_(m.all_weights[1][0])
                nn.init.orthogonal_(m.all_weights[1][1])
                logging.debug("Initialized LSTM layer: %s", str(m))

    def initialize_weights_from_pretrained(self, pretrained_net_fname):
        """Initialize module weights from a pretrained model.

        Parameters
        ----------
        pretrained_net_fname : str
            Path to the pretrained model file (e.g. 'checkpoint.pth')
        """
        logging.debug("Loading pretrained weights from: %s", pretrained_net_fname)
        pretrained_dict = torch.load(pretrained_net_fname)
        net_state_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(pretrained_dict)
        self.load_state_dict(net_state_dict)
        logging.info("Successfully loaded pretrained weights from: %s", pretrained_net_fname)

    def load(self, path):
        """Load module weights from a saved model file.

        Parameters
        ----------
        path : str
            Path to the saved model file
        """
        logging.debug("Loading model from: %s", path)
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        logging.info("Successfully loaded model from: %s", path)

    def save(self, fname=None):
        """Save module weights to a file.

        Parameters
        ----------
        fname : str, optional
            Path to save the model. If None, generates a timestamped filename.

        Returns
        -------
        str
            Path to the saved model file
        """
        if fname is None:
            fname = time.strftime("model" + '%m%d_%H:%M:%S.pth')
        logging.debug("Saving model to: %s", fname)
        torch.save(self.state_dict(), fname)
        logging.info("Successfully saved model to: %s", fname)
        return fname

    def test(self, input_size):
        """Test the module with dummy input.

        Parameters
        ----------
        input_size : tuple
            Shape of the input tensor to test with

        Notes
        -----
        This method runs a forward pass with zero tensor and logs shapes.
        """
        logging.debug("Testing model with input size: %s", str(input_size))
        device = list(self.parameters)[0].device
        x = torch.zeros(input_size).to(device)
        out = self.forward(x)
        logging.info("Test complete - Input shape: %s, Output shape: %s", 
                    str(input_size), str(out.shape))


class Flatten(nn.Module):
    """Flatten module that reshapes input tensor to (batch_size, -1).

    This module is commonly used between convolutional and linear layers
    to flatten the feature maps into a 1D vector.
    """
    def forward(self, x):
        """Reshape input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of any shape

        Returns
        -------
        torch.Tensor
            Flattened tensor of shape (batch_size, -1)
        """
        logging.debug("Flattening tensor of shape: %s", str(x.shape))
        out = x.view(x.size(0), -1)
        logging.debug("Flattened tensor shape: %s", str(out.shape))
        return out


class EXP(nn.Module):
    """Exponential activation module.

    Applies element-wise exponential function to the input tensor.
    """
    def forward(self, x):
        """Apply exponential function.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Exponential of input tensor
        """
        logging.debug("Computing exp of tensor with shape: %s", str(x.shape))
        return x.exp()


class Residual(nn.Module):
    """Residual connection module.

    Adds the output of a transformation to its input.
    """
    def __init__(self, fn):
        """
        Parameters
        ----------
        fn : callable
            Transformation function to apply
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        """Apply residual connection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        **kwargs : dict
            Additional arguments passed to fn

        Returns
        -------
        torch.Tensor
            Sum of input and transformed tensor
        """
        logging.debug("Applying residual connection to tensor of shape: %s", str(x.shape))
        return self.fn(x, **kwargs) + x


class BasicConv1d(BasicModule):
    """Basic 1D convolutional module for genomic sequence processing.

    Combines convolution, batch normalization, activation, dropout, and pooling
    in a configurable sequence. Designed for processing genomic sequence data.

    Parameters
    ----------
    in_planes : int
        Number of input channels
    out_planes : int
        Number of output channels
    kernel_size : int, optional
        Size of the convolution kernel (default: 3)
    conv_args : dict, optional
        Additional arguments for convolution layer
    bn : bool, optional
        Whether to use batch normalization (default: True)
    activation : nn.Module, optional
        Activation function to use (default: nn.ReLU)
    activation_args : dict, optional
        Arguments for activation function
    dropout : bool, optional
        Whether to use dropout (default: True)
    dropout_args : dict, optional
        Arguments for dropout layer (default: {'p': 0.5})
    pool : nn.Module, optional
        Pooling layer to use (default: nn.AvgPool1d)
    pool_args : dict, optional
        Arguments for pooling layer (default: {'kernel_size': 3})

    Notes
    -----
    The module applies operations in the following order:
    1. Convolution
    2. Batch Normalization (optional)
    3. Activation (optional)
    4. Dropout (optional)
    5. Pooling (optional)
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, conv_args={}, 
                    bn=True, 
                    activation=nn.ReLU, activation_args={}, 
                    dropout=True, dropout_args={'p':0.5},
                    pool=nn.AvgPool1d, pool_args={'kernel_size': 3}):
        super().__init__()
        self.in_channels = in_planes
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, **conv_args)
        self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.activation = activation(**activation_args) if activation is not None else None
        self.dropout = nn.Dropout(**dropout_args) if dropout else None
        self.pool = pool(**pool_args) if pool else None

    def forward(self, x):
        """Forward pass of the convolutional module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, sequence_length)

        Returns
        -------
        torch.Tensor
            Processed tensor
        """
        logging.debug("BasicConv1d input shape: %s", str(x.shape))
        
        x = self.conv(x)
        logging.debug("After conv shape: %s", str(x.shape))
        
        if self.bn is not None:
            x = self.bn(x)
            logging.debug("After batch norm shape: %s", str(x.shape))
            
        if self.activation is not None:
            x = self.activation(x)
            logging.debug("After activation shape: %s", str(x.shape))
            
        if self.dropout is not None:
            x = self.dropout(x)
            logging.debug("After dropout shape: %s", str(x.shape))
            
        if self.pool is not None:
            x = self.pool(x)
            logging.debug("After pooling shape: %s", str(x.shape))
            
        return x


class BasicRNNModule(BasicModule):
    """Basic RNN/LSTM module for sequential genomic data.

    Implements a bidirectional LSTM layer with batch-first processing,
    suitable for analyzing sequential patterns in genomic data.

    Parameters
    ----------
    LSTM_input_size : int, optional
        Size of input features (default: 512)
    LSTM_hidden_size : int, optional
        Size of hidden state (default: 512)
    LSTM_hidden_layers : int, optional
        Number of LSTM layers (default: 2)

    Notes
    -----
    - Uses batch_first=True for easier handling of variable length sequences
    - Implements bidirectional LSTM for capturing both forward and reverse patterns
    """
    def __init__(self, LSTM_input_size=512, LSTM_hidden_size=512, LSTM_hidden_layes=2):
        super().__init__()
        self.rnn_hidden_state = None
        self.rnn = nn.LSTM(
            input_size=LSTM_input_size,
            hidden_size=LSTM_hidden_size,
            num_layers=LSTM_hidden_layes,
            batch_first=True,  # batch, seq, feature
            bidirectional=True,
        )
        logging.debug("Initialized LSTM with input_size=%d, hidden_size=%d, num_layers=%d",
                     LSTM_input_size, LSTM_hidden_size, LSTM_hidden_layes)

    def forward(self, input):
        """Forward pass of the RNN module.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, 2*hidden_size)
        """
        logging.debug("BasicRNNModule input shape: %s", str(input.shape))
        output, self.rnn_hidden_state = self.rnn(input, None)
        logging.debug("LSTM output shape: %s", str(output.shape))
        return output


class BasicLinearModule(BasicModule):
    """Basic linear transformation module.

    Implements a linear layer with optional batch normalization, activation,
    and dropout. Used for feature transformation and dimensionality reduction.

    Parameters
    ----------
    input_size : int
        Size of input features
    output_size : int
        Size of output features
    bias : bool, optional
        Whether to include bias in linear layer (default: True)
    bn : bool, optional
        Whether to use batch normalization (default: True)
    activation : nn.Module, optional
        Activation function to use (default: nn.ReLU)
    activation_args : dict, optional
        Arguments for activation function
    dropout : bool, optional
        Whether to use dropout (default: True)
    dropout_args : dict, optional
        Arguments for dropout layer (default: {'p': 0.5})

    Notes
    -----
    The module applies operations in the following order:
    1. Linear transformation
    2. Batch Normalization (optional)
    3. Activation (optional)
    4. Dropout (optional)
    """
    def __init__(self, input_size, output_size, bias=True, bn=True, 
                    activation=nn.ReLU, activation_args={}, 
                    dropout=True, dropout_args={'p':0.5}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size, bias=bias)
        self.bn = nn.BatchNorm1d(output_size, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.activation = activation(**activation_args) if activation is not None else None
        self.dropout = nn.Dropout(**dropout_args) if dropout else None

    def forward(self, x):
        """Forward pass of the linear module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size)

        Returns
        -------
        torch.Tensor
            Transformed tensor of shape (batch_size, output_size)
        """
        logging.debug("BasicLinearModule input shape: %s", str(x.shape))
        
        x = self.linear(x)
        logging.debug("After linear shape: %s", str(x.shape))
        
        if self.bn is not None:
            x = self.bn(x)
            logging.debug("After batch norm shape: %s", str(x.shape))
            
        if self.activation is not None:
            x = self.activation(x)
            logging.debug("After activation shape: %s", str(x.shape))
            
        if self.dropout is not None:
            x = self.dropout(x)
            logging.debug("After dropout shape: %s", str(x.shape))
            
        return x


class BasicPredictor(BasicModule):
    """Task-specific prediction head module.

    Implements different types of prediction heads for various tasks:
    - Binary classification (sigmoid activation)
    - Multi-class classification (softmax activation)
    - Regression (identity activation)

    Parameters
    ----------
    input_size : int
        Size of input features
    output_size : int
        Number of output predictions
    tasktype : str, optional
        Type of prediction task (default: 'binary_classification')
        Options: ['none', 'binary_classification', 'classification', 'regression']

    Notes
    -----
    The predictor can be switched between different task types during training
    using the switch_task method.
    """
    def __init__(self, input_size, output_size, tasktype='binary_classification'):
        super().__init__()
        self.supported_tasks = ['none', 'binary_classification', 'classification', 'regression']
        logging.debug("Initializing BasicPredictor for task type: %s", tasktype)

        self.input_size = input_size
        self.tasktype = tasktype
        
        self.Map = nn.Linear(input_size, output_size, bias=True)
        self.switch_task(tasktype)

    def forward(self, x):
        """Forward pass of the predictor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size)

        Returns
        -------
        torch.Tensor
            Predictions of shape (batch_size, output_size)
        """
        logging.debug("BasicPredictor input shape: %s", str(x.shape))
        mapped = self.Map(x)
        logging.debug("After mapping shape: %s", str(mapped.shape))
        out = self.Pred(mapped)
        logging.debug("Prediction output shape: %s", str(out.shape))
        return out

    def switch_task(self, tasktype):
        """Switch the predictor to a different task type.

        Parameters
        ----------
        tasktype : str
            New task type to switch to
        """
        msg = 'tasktype: %s not supported, check the document' % tasktype
        assert tasktype in self.supported_tasks, msg
        logging.debug("Switching task type to: %s", tasktype)

        if tasktype == 'none':
            self.Map = nn.Sequential()
            self.Pred = nn.Sequential()
        elif tasktype == 'classification':
            self.Pred = nn.Softmax(dim=1)
        elif tasktype == 'binary_classification':
            self.Pred = nn.Sigmoid()
        elif tasktype == 'regression':
            self.Pred = nn.Sequential()

        self.tasktype = tasktype
        logging.info("Successfully switched to task type: %s", tasktype)

    def current_task(self):
        """Get the current task type.

        Returns
        -------
        str
            Current task type
        """
        return self.tasktype

    def remove(self):
        """Remove the predictor by switching to 'none' task type."""
        logging.debug("Removing predictor functionality")
        self.switch_task('none')


class BasicLoss(nn.Module):
    """Task-specific loss function module.

    Provides appropriate loss functions for different prediction tasks:
    - Binary classification: BCELoss
    - Multi-class classification: CrossEntropyLoss
    - Regression: MSELoss

    Parameters
    ----------
    tasktype : str, optional
        Type of prediction task (default: 'binary_classification')
    reduction : str, optional
        Reduction method for the loss (default: 'mean')
        Options: ['none', 'mean', 'sum']

    Notes
    -----
    The loss function can be switched between different task types during
    training using the switch_task method.
    """
    def __init__(self, tasktype='binary_classification', reduction='mean'):
        super().__init__()
        self.supported_tasks = ['binary_classification', 'classification', 'regression']
        logging.debug("Initializing BasicLoss for task type: %s", tasktype)

        self.tasktype = tasktype
        self.reduction = reduction

        self.switch_task(tasktype)

    def forward(self, pred, target):
        """Compute the loss.

        Parameters
        ----------
        pred : torch.Tensor
            Model predictions
        target : torch.Tensor
            Ground truth targets

        Returns
        -------
        torch.Tensor
            Computed loss value
        """
        logging.debug("Computing loss for shapes - pred: %s, target: %s",
                     str(pred.shape), str(target.shape))
        return self.loss(pred, target)

    def switch_task(self, tasktype):
        """Switch to a different loss function.

        Parameters
        ----------
        tasktype : str
            New task type to switch to
        """
        msg = 'tasktype: %s not supported, check the document' % tasktype
        assert tasktype in self.supported_tasks, msg
        logging.debug("Switching loss function to: %s", tasktype)

        if tasktype == 'classification':
            self.loss = nn.CrossEntropyLoss(reduction=self.reduction)
        elif tasktype == 'binary_classification':
            self.loss = nn.BCELoss(reduction=self.reduction)
        elif tasktype == 'regression':
            self.loss = nn.MSELoss(reduction=self.reduction)

        self.tasktype = tasktype
        logging.info("Successfully switched to loss function for: %s", tasktype)

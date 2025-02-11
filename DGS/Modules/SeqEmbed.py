"""Sequence Embedding Modules for genomic sequence analysis.

This module provides various embedding methods for transforming one-hot encoded
DNA sequences into learned vector representations. It includes:

Classes
-------
BasicConvEmbed
    Basic convolutional embedding that applies 1D convolution to sequences

RevCompConvEmbed
    Convolutional embedding that considers both forward and reverse complement
    sequences, useful for strand-agnostic analysis

CharConvModule
    Wide and shallow character-level convolution module with multiple kernel sizes,
    designed for capturing sequence motifs at different scales

Notes
-----
All embedding modules expect input sequences in the form of one-hot encoded tensors
with shape (batch_size, 4, sequence_length) where 4 represents the nucleotides (A,C,G,T).
Debug-level logging is available for tracking tensor shapes through the network.
"""

# Code:   jiaqili@zju.edu

import logging

import torch
import torch.nn as nn

__all__ = ["BasicConvEmbed", "RevComp", "RevCompConvEmbed", "CharConvModule"]

class BasicConvEmbed(nn.Module):
    """Basic convolutional embedding module for DNA sequences.

    Embeds sequences using a convolutional layer with optional batch normalization,
    activation, and pooling.

    Parameters
    ----------
    out_planes : int
        Number of output channels (embedding dimension)
    kernel_size : int, optional
        Size of the convolution kernel (default: 3)
    in_planes : int, optional
        Number of input channels, typically 4 for DNA sequences (default: 4)
    conv_args : dict, optional
        Additional arguments for convolution layer (default: {'stride':1, 'padding':0,
        'dilation':1, 'groups':1})
    bn : bool, optional
        Whether to use batch normalization (default: False)
    activation : nn.Module, optional
        Activation function to use (default: nn.ReLU)
    activation_args : dict, optional
        Arguments for activation function (default: {})
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
    4. Pooling (optional)
    """
    def __init__(self, out_planes, kernel_size=3, in_planes=4, 
                    conv_args={'stride':1, 'padding':0, 'dilation':1, 'groups':1}, 
                    bn=False, activation=nn.ReLU, activation_args={}, 
                    pool=nn.AvgPool1d, pool_args={'kernel_size': 3}):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, 
                        **conv_args)
        self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.activation = activation(**activation_args) if activation else None
        self.pool = pool(**pool_args) if pool else None

    def forward(self, x):
        """Forward pass of the embedding module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 4, sequence_length)

        Returns
        -------
        torch.Tensor
            Embedded sequence tensor
        """
        logging.debug("BasicConvEmbed input shape: %s", str(x.shape))
        
        x = self.conv(x)
        logging.debug("After convolution shape: %s", str(x.shape))
        
        if self.bn is not None:
            x = self.bn(x)
            logging.debug("After batch norm shape: %s", str(x.shape))
            
        if self.activation is not None:
            x = self.activation(x)
            logging.debug("After activation shape: %s", str(x.shape))
            
        if self.pool is not None:
            x = self.pool(x)
            logging.debug("After pooling shape: %s", str(x.shape))
            
        return x


class RevComp(nn.Module):
    """Reverse complement module for DNA sequences.

    Computes the reverse complement of one-hot encoded DNA sequences by flipping
    both the channel (base) and sequence dimensions.
    """
    def forward(self, x):
        """Compute reverse complement.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 4, sequence_length) or
            (batch_size, 4, 1, sequence_length)

        Returns
        -------
        torch.Tensor
            Reverse complemented sequence tensor
        """
        if len(x.shape) == 3:
            return x.flip([1, 2])  # (batchsize, 4, seqlen)
        elif len(x.shape) == 4:
            return x.flip([1, -1])  # (batchsize, 4, 1, seqlen)


class RevCompConvEmbed(nn.Module):
    """Reverse complement aware convolutional embedding module.

    Applies convolutional embedding to both forward and reverse complement
    sequences and combines their features. This makes the embedding invariant
    to strand orientation.

    Parameters
    ----------
    out_planes : int
        Number of output channels (embedding dimension)
    kernel_size : int, optional
        Size of the convolution kernel (default: 3)
    in_planes : int, optional
        Number of input channels, typically 4 for DNA sequences (default: 4)
    conv_args : dict, optional
        Additional arguments for convolution layer
    bn : bool, optional
        Whether to use batch normalization (default: False)
    activation : nn.Module, optional
        Activation function to use (default: nn.ReLU)
    activation_args : dict, optional
        Arguments for activation function (default: {})
    pool : nn.Module, optional
        Pooling layer to use (default: nn.AvgPool1d)
    pool_args : dict, optional
        Arguments for pooling layer (default: {'kernel_size': 3})

    Notes
    -----
    The module:
    1. Computes reverse complement of input
    2. Applies same embedding to both sequences
    3. Combines features by addition
    """
    def __init__(self, out_planes, kernel_size=3, in_planes=4, 
                    conv_args={'stride':1, 'padding':0, 'dilation':1, 'groups':1}, 
                    bn=False, activation=nn.ReLU, activation_args={}, 
                    pool=nn.AvgPool1d, pool_args={'kernel_size': 3}):
        super().__init__()
        self.RevCompConvEmbed = BasicConvEmbed(out_planes, kernel_size=kernel_size, 
                    in_planes=in_planes, conv_args=conv_args, 
                    bn=bn, activation=activation, activation_args=activation_args, 
                    pool=pool, pool_args=pool_args)
        self.RevComp = RevComp()

    def forward(self, x):
        """Forward pass computing strand-agnostic features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 4, sequence_length)

        Returns
        -------
        torch.Tensor
            Combined features from both strands
        """
        logging.debug("RevCompConvEmbed input shape: %s", str(x.shape))
        
        fmap1 = self.RevCompConvEmbed(x)
        logging.debug("Forward strand features shape: %s", str(fmap1.shape))
        
        fmap2 = self.RevCompConvEmbed(self.RevComp(x))
        logging.debug("Reverse strand features shape: %s", str(fmap2.shape))
        
        combined = fmap1 + fmap2
        logging.debug("Combined features shape: %s", str(combined.shape))
        return combined


class CharConvModule(nn.Module):
    """Wide and shallow character-level convolution module.

    Applies multiple parallel convolutions with different kernel sizes to capture
    sequence patterns at different scales. Features from all convolutions are
    concatenated to form the final representation.

    Parameters
    ----------
    numFiltersConv1 : int, optional
        Number of filters for first convolution (default: 40)
    filterLenConv1 : int, optional
        Kernel size for first convolution (default: 5)
    numFiltersConv2 : int, optional
        Number of filters for second convolution (default: 44)
    filterLenConv2 : int, optional
        Kernel size for second convolution (default: 15)
    numFiltersConv3 : int, optional
        Number of filters for third convolution (default: 44)
    filterLenConv3 : int, optional
        Kernel size for third convolution (default: 25)
    bn : bool, optional
        Whether to use batch normalization (default: False)
    activation : nn.Module, optional
        Activation function to use (default: nn.ReLU)
    activation_args : dict, optional
        Arguments for activation function (default: {})
    pool : nn.Module, optional
        Pooling layer to use (default: nn.AvgPool1d)
    pool_args : dict, optional
        Arguments for pooling layer (default: {'kernel_size': 3})

    Notes
    -----
    The module:
    1. Applies three parallel convolutions with different kernel sizes
    2. Concatenates features from all convolutions
    3. Applies optional batch norm, activation, and pooling
    """
    def __init__(self, numFiltersConv1=40, filterLenConv1=5,
                        numFiltersConv2=44, filterLenConv2=15,
                        numFiltersConv3=44, filterLenConv3=25,
                        bn=False, activation=nn.ReLU, activation_args={}, 
                        pool=nn.AvgPool1d, pool_args={'kernel_size': 3}):
        
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=numFiltersConv1, 
                                kernel_size=filterLenConv1, 
                                padding=(filterLenConv1 - 1) // 2)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=numFiltersConv2, 
                                kernel_size=filterLenConv2, 
                                padding=(filterLenConv2 - 1) // 2)
        self.conv3 = nn.Conv1d(in_channels=4, out_channels=numFiltersConv3, 
                                kernel_size=filterLenConv3, 
                                padding=(filterLenConv3 - 1) // 2)

        out_planes = numFiltersConv1 + numFiltersConv2 + numFiltersConv3
        self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.activation = activation(**activation_args) if activation is not None else None
        self.pool = pool(**pool_args) if pool else None

    def forward(self, x):
        """Forward pass applying parallel convolutions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 4, sequence_length)

        Returns
        -------
        torch.Tensor
            Concatenated features from all convolutions
        """
        logging.debug("CharConvModule input shape: %s", str(x.shape))

        out1 = self.conv1(x)
        logging.debug("Conv1 output shape: %s", str(out1.shape))
        
        out2 = self.conv2(x)
        logging.debug("Conv2 output shape: %s", str(out2.shape))

        out3 = self.conv3(x)
        logging.debug("Conv3 output shape: %s", str(out3.shape))
        
        out = torch.cat([out1, out2, out3], dim=1)
        logging.debug("Concatenated output shape: %s", str(out.shape))

        if self.bn is not None:
            out = self.bn(out)
            logging.debug("After batch norm shape: %s", str(out.shape))
            
        if self.activation is not None:
            out = self.activation(out)
            logging.debug("After activation shape: %s", str(out.shape))
            
        if self.pool is not None:
            out = self.pool(out)
            logging.debug("After pooling shape: %s", str(out.shape))

        return out


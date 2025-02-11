"""Convolutional Models in DGS.

This module provides convolutional neural network architectures for genomic sequence analysis:

Classes
-------
CNN
    Basic Convolutional Neural Network model that extends BasicModel.
    Uses standard convolution layers for sequence processing.

CAN
    Convolutional Attention Network model that extends BasicModel.
    Incorporates CBAM (Convolutional Block Attention Module) for attention-based sequence processing.
"""

# Code:   jiaqili@zju.edu

from collections import OrderedDict

import torch
from torch import nn

from .BasicModel import BasicModel
from ..Modules import BasicConvEmbed, RevCompConvEmbed, CharConvModule
from ..Modules import BasicConv1d, Flatten, BasicLinearModule, BasicPredictor
from ..Modules import CBAM


class CNN(BasicModel):
    """Convolutional Neural Network model for genomic sequence analysis.

    A basic CNN architecture that extends BasicModel with convolutional layers.
    The model consists of:
    1. Embedding layer with convolution
    2. Encoder with convolution and global average pooling
    3. Decoder with linear transformation
    4. Task-specific predictor

    Parameters
    ----------
    output_size : int
        Size of the output layer (number of prediction targets)
    out_planes : int, optional
        Number of output channels in the first convolution layer (default: 128)
    kernel_size : int, optional
        Size of the convolution kernel (default: 3)
    in_planes : int, optional
        Number of input channels, typically 4 for DNA sequences (default: 4)
    conv_args : dict, optional
        Arguments for convolution layers (default: {'stride':1, 'padding':0, 
        'dilation':1, 'groups':1, 'bias':True})
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
    tasktype : str, optional
        Type of task: 'regression' or 'classification' (default: 'regression')

    Notes
    -----
    - Input sequences should be one-hot encoded with shape (batch_size, 4, sequence_length)
    - The model uses a fixed architecture with convolution -> pooling -> linear layers
    - Global average pooling is used to handle variable length sequences
    """
    def __init__(self, output_size, 
                    out_planes=128, kernel_size=3, in_planes=4, 
                    conv_args={'stride':1, 'padding':0, 'dilation':1, 'groups':1, 'bias':True}, 
                    bn=False, activation=nn.ReLU, activation_args={}, 
                    pool=nn.AvgPool1d, pool_args={'kernel_size': 3},
                    tasktype='regression'):
        super().__init__()
        self.Embedding = BasicConvEmbed(out_planes=out_planes, 
                    kernel_size=kernel_size, in_planes=in_planes, conv_args=conv_args, 
                    bn=bn, activation=activation, activation_args=activation_args, 
                    pool=pool, pool_args=pool_args)
        self.Encoder = nn.Sequential(OrderedDict([
                        ('Conv', BasicConv1d(in_planes=out_planes, out_planes=256)),
                        ('GAP', nn.AdaptiveAvgPool1d(8)),
                        ('Flatten', Flatten())
                        ]))
        self.Decoder = BasicLinearModule(256 * 8, 256)
        self.Predictor = BasicPredictor(256, output_size, tasktype=tasktype)


class CAN(BasicModel):
    """Convolutional Attention Network for genomic sequence analysis.

    An advanced CNN architecture that incorporates attention mechanisms using CBAM
    (Convolutional Block Attention Module). The model consists of:
    1. Embedding layer with convolution
    2. Encoder with convolution, attention (CBAM), and global average pooling
    3. Decoder with linear transformation
    4. Task-specific predictor

    Parameters
    ----------
    output_size : int
        Size of the output layer (number of prediction targets)
    out_planes : int, optional
        Number of output channels in the first convolution layer (default: 128)
    kernel_size : int, optional
        Size of the convolution kernel (default: 3)
    in_planes : int, optional
        Number of input channels, typically 4 for DNA sequences (default: 4)
    conv_args : dict, optional
        Arguments for convolution layers (default: {'stride':1, 'padding':0, 
        'dilation':1, 'groups':1, 'bias':False})
    bn : bool, optional
        Whether to use batch normalization (default: True)
    activation : nn.Module, optional
        Activation function to use (default: nn.ReLU)
    activation_args : dict, optional
        Arguments for activation function (default: {})
    pool : nn.Module, optional
        Pooling layer to use (default: nn.AvgPool1d)
    pool_args : dict, optional
        Arguments for pooling layer (default: {'kernel_size': 3})
    tasktype : str, optional
        Type of task: 'regression' or 'classification' (default: 'regression')

    Notes
    -----
    - Input sequences should be one-hot encoded with shape (batch_size, 4, sequence_length)
    - CBAM attention helps the model focus on important regions of the sequence
    - The model uses batch normalization by default for better training stability
    - Global average pooling is used to handle variable length sequences

    References
    ----------
    .. [1] Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). 
           CBAM: Convolutional Block Attention Module. 
           In Proceedings of the European Conference on Computer Vision (ECCV)
    """
    def __init__(self, output_size, 
                    out_planes=128, kernel_size=3, in_planes=4, 
                    conv_args={'stride':1, 'padding':0, 'dilation':1, 'groups':1, 'bias':False}, 
                    bn=True, activation=nn.ReLU, activation_args={}, 
                    pool=nn.AvgPool1d, pool_args={'kernel_size': 3},
                    tasktype='regression'):
        super().__init__()
        self.Embedding = BasicConvEmbed(out_planes=out_planes, 
                    kernel_size=kernel_size, in_planes=in_planes, conv_args=conv_args, 
                    bn=bn, activation=activation, activation_args=activation_args, 
                    pool=pool, pool_args=pool_args)
        self.Encoder = nn.Sequential(OrderedDict([
                        ('Conv', BasicConv1d(in_planes=128, out_planes=256)),
                        ('Attention', CBAM(256)),
                        ('GAP', nn.AdaptiveAvgPool1d(8)),
                        ('Flatten', Flatten())
                        ]))
        self.Decoder = BasicLinearModule(256 * 8, 256)
        self.Predictor = BasicPredictor(256, output_size, tasktype=tasktype)



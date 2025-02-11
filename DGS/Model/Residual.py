"""Residual Network architectures for genomic sequence analysis.

This module implements ResNet-style architectures adapted for genomic sequence data.
The implementation includes attention mechanisms (CBAM) and is based on the following papers:

References
----------
.. [1] Li, X., et al. (2019). Selective Kernel Networks.
       IEEE Conference on Computer Vision and Pattern Recognition.

.. [2] Li, X., et al. (2019). Spatial Group-wise Enhance: Enhancing Semantic 
       Feature Learning in Convolutional Networks.
       arXiv preprint arXiv:1905.09646.

.. [3] Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module.
       European Conference on Computer Vision (ECCV).

Notes
-----
The implementation is modified from the original ResNet to handle one-hot encoded
DNA sequences and includes attention modules using CBAM (Convolutional Block 
Attention Module).
"""

# Code:   https://github.com/implus/PytorchInsight/blob/master/classification/models/imagenet/resnet_cbam.py
# Note:   modified for onehot sequence input
#         add attention module using CBAM

__all__  = ["BasicBlock", "Bottleneck", "ResNet", "ResidualNet"]

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from ..Modules.BasicModule import BasicModule

class BasicConv(nn.Module):
    """Basic convolution module with optional batch normalization and activation.

    Parameters
    ----------
    in_planes : int
        Number of input channels
    out_planes : int
        Number of output channels
    kernel_size : int or tuple
        Size of the convolution kernel
    stride : int, optional
        Stride of convolution (default: 1)
    padding : int, optional
        Padding size (default: 0)
    dilation : int, optional
        Dilation rate (default: 1)
    groups : int, optional
        Number of groups for grouped convolution (default: 1)
    relu : bool, optional
        Whether to include ReLU activation (default: True)
    bn : bool, optional
        Whether to include batch normalization (default: True)
    bias : bool, optional
        Whether to include bias in convolution (default: False)
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, 
                        dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                                stride=stride, padding=padding, dilation=dilation,
                                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        """Forward pass of the convolution module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output after convolution, optional batch norm and activation
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    """Flattens input tensor except batch dimension."""
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    """Channel attention module of CBAM.

    Applies channel-wise attention using both max and average pooling
    information. The attention weights are computed using a multi-layer
    perceptron.

    Parameters
    ----------
    gate_channels : int
        Number of input channels
    reduction_ratio : int, optional
        Reduction ratio for the bottleneck (default: 16)
    pool_types : list of str, optional
        Types of pooling to use (default: ['avg', 'max'])
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass of channel attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)

        Returns
        -------
        torch.Tensor
            Channel-attended tensor of same shape as input
        """
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = self.avgpool(x)
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = self.maxpool(x)
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = self.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    """Channel pooling module that concatenates max and mean pooling results."""
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    """Spatial attention module of CBAM.

    Applies spatial attention using both channel-wise max and average pooling
    information. The attention weights are computed using a convolution layer.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the convolution kernel for computing spatial attention (default: 7)
    """
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, (1, kernel_size), stride=1, padding=(0, (kernel_size-1) // 2), relu=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        """Forward pass of spatial attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)

        Returns
        -------
        torch.Tensor
            Spatially-attended tensor of same shape as input
        """
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).

    Combines channel and spatial attention mechanisms to enhance feature
    representation. The module first applies channel attention followed by
    spatial attention.

    Parameters
    ----------
    gate_channels : int
        Number of input channels
    reduction_ratio : int, optional
        Reduction ratio for channel attention (default: 16)
    pool_types : list of str, optional
        Types of pooling for channel attention (default: ['avg', 'max'])
    no_spatial : bool, optional
        Whether to exclude spatial attention (default: False)
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        """Forward pass applying channel and optional spatial attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)
            or (batch_size, 4, sequence_length) for DNA sequences

        Returns
        -------
        torch.Tensor
            Attended tensor of same shape as input
        """
        if len(x.shape) == 3 and x.shape[1] == 4: # (B, 4, L)
            x = x.unsqueeze(2) # (B, 4, 1, L) update 2D sequences
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

def conv1x3(in_planes, out_planes, stride=1):
    """1x3 convolution with padding for DNA sequences."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), stride=stride,
                     padding=(0,1), bias=False)

class BasicBlock(nn.Module):
    """Basic residual block for ResNet.

    A basic block consists of two 1x3 convolutions with batch normalization,
    ReLU activation, and optional CBAM attention. The block includes a residual
    connection that adds the input to the transformed features.

    Parameters
    ----------
    inplanes : int
        Number of input channels
    planes : int
        Number of output channels
    stride : int, optional
        Stride for convolution (default: 1)
    downsample : nn.Module, optional
        Downsampling module for residual connection (default: None)
    use_cbam : bool, optional
        Whether to use CBAM attention (default: False)

    Notes
    -----
    The block follows this architecture:
    1. Conv1x3 -> BatchNorm -> ReLU
    2. Conv1x3 -> BatchNorm
    3. CBAM (optional)
    4. Add residual connection
    5. ReLU
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        """Forward pass of the basic block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor after residual block processing
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNet.

    A bottleneck block consists of three convolutions: 1x1, 1x3, and 1x1.
    This design reduces parameters while maintaining model capacity. The block
    includes batch normalization, ReLU activation, and optional CBAM attention.

    Parameters
    ----------
    inplanes : int
        Number of input channels
    planes : int
        Number of intermediate channels (expanded by 4 in the last conv)
    stride : int, optional
        Stride for convolution (default: 1)
    downsample : nn.Module, optional
        Downsampling module for residual connection (default: None)
    use_cbam : bool, optional
        Whether to use CBAM attention (default: False)

    Notes
    -----
    The block follows this architecture:
    1. Conv1x1 -> BatchNorm -> ReLU
    2. Conv1x3 -> BatchNorm -> ReLU
    3. Conv1x1 -> BatchNorm
    4. CBAM (optional)
    5. Add residual connection
    6. ReLU
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(1,3), stride=stride,
                               padding=(0,1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes * 4, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        """Forward pass of the bottleneck block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor after bottleneck block processing
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(BasicModule):
    """ResNet architecture adapted for genomic sequence analysis.

    This implementation modifies the standard ResNet architecture to handle
    one-hot encoded DNA sequences. It supports different network configurations
    (18, 34, 50, 101 layers) and includes optional attention mechanisms.

    Parameters
    ----------
    block : type
        Type of residual block to use (BasicBlock or Bottleneck)
    layers : list of int
        Number of blocks in each layer
    network_type : str, optional
        Type of network configuration ('ImageNet', 'CIFAR10', 'CIFAR100', None)
    num_classes : int, optional
        Number of output classes (default: None)
    att_type : str, optional
        Type of attention to use (default: None)

    Notes
    -----
    - Input shape: (batch_size, 4, sequence_length) for DNA sequences
    - The network type determines the initial convolution and pooling layers
    - Supports CBAM attention mechanism
    - Uses batch normalization and ReLU activation throughout
    """
    
    def __init__(self, block, layers, network_type=None, num_classes=None, att_type=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR 
        if network_type is None:
            self.conv1 = None
        elif network_type == "ImageNet":
            self.conv1 = nn.Conv2d(4, 64, kernel_size=(1, 7), stride=1, padding=(0,3), bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=(0,1))
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        elif network_type == "CIFAR10" or network_type == "CIFAR100" :
            self.conv1 = nn.Conv2d(4, 64, kernel_size=(1, 3), stride=1, padding=(0,1), bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        if att_type=='CBAM':
            self.bam1 = CBAM(64*block.expansion)
            self.bam2 = CBAM(128*block.expansion)
            self.bam3 = CBAM(256*block.expansion)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        self.layer1 = self._make_layer(block, 64,  layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1, att_type=att_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, att_type=att_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, att_type=att_type)
        
        if num_classes is not None:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc = None

        init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        """Create a layer of residual blocks.

        Parameters
        ----------
        block : type
            Type of residual block to use
        planes : int
            Number of output channels
        blocks : int
            Number of blocks in the layer
        stride : int, optional
            Stride for first block (default: 1)
        att_type : str, optional
            Type of attention to use (default: None)

        Returns
        -------
        nn.Sequential
            Sequential container of residual blocks
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type=='CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of the ResNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 4, sequence_length)

        Returns
        -------
        torch.Tensor
            Output tensor, shape depends on network configuration
        """
        if self.network_type is not None: # which should embed sequence directly
            if len(x.shape) == 3: # (B, 4, L)
                x = x.unsqueeze(2) # (B, 4, 1, L) update 2D sequences
            
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)
        if not self.bam1 is None:
            x = self.bam1(x)

        x = self.layer2(x)
        if not self.bam2 is None:
            x = self.bam2(x)

        x = self.layer3(x)
        if not self.bam3 is None:
            x = self.bam3(x)

        x = self.layer4(x)

        if self.network_type == "ImageNet":
            x = self.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)

        if not self.fc is None:
            x = self.fc(x)

        return x
        

def ResidualNet(network_type, depth, num_classes, att_type):
    """Factory function to create ResNet models.

    Creates a ResNet model with specified depth and configuration.

    Parameters
    ----------
    network_type : str
        Type of network ('ImageNet', 'CIFAR10', 'CIFAR100')
    depth : int
        Depth of the network (18, 34, 50, or 101)
    num_classes : int
        Number of output classes
    att_type : str
        Type of attention to use

    Returns
    -------
    ResNet
        Configured ResNet model

    Raises
    ------
    AssertionError
        If network_type or depth is not supported
    """
    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type)

    return model


"""Convolutional Block Attention Module (CBAM) for genomic sequences.

This module implements a modified version of CBAM adapted for genomic sequence data.
CBAM enhances feature representations by applying both channel and spatial attention
mechanisms sequentially.

Classes
-------
BasicConv
    Basic convolution with optional batch normalization and activation
ChannelGate
    Channel attention mechanism using max and average pooling
SpatialGate
    Spatial attention mechanism using channel pooling
CBAM
    Main attention module combining channel and spatial attention

References
----------
.. [1] Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018).
       CBAM: Convolutional Block Attention Module.
       Proceedings of the European Conference on Computer Vision (ECCV).

Notes
-----
This implementation is modified from the original CBAM to work with 1D genomic
sequences. The input is expected to be of shape (batch_size, channels, sequence_length)
or will be automatically reshaped to (batch_size, channels, 1, sequence_length).
"""

import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CBAM"]

class BasicConv(nn.Module):
    """Basic convolution module with optional batch normalization and activation.

    Parameters
    ----------
    in_planes : int
        Number of input channels
    out_planes : int
        Number of output channels
    kernel_size : int or tuple
        Size of convolution kernel
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
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, 
                            stride=stride, padding=padding, dilation=dilation,
                            groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        """Forward pass of convolution module.

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
        logging.debug("Conv output shape: %s", str(x.shape))
        
        if self.bn is not None:
            x = self.bn(x)
            logging.debug("BatchNorm output shape: %s", str(x.shape))
            
        if self.relu is not None:
            x = self.relu(x)
            logging.debug("ReLU output shape: %s", str(x.shape))
            
        return x

class Flatten(nn.Module):
    """Flattens input tensor except batch dimension."""
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    """Channel attention module.

    Applies channel-wise attention using both max and average pooling information.
    The attention weights are computed using a multi-layer perceptron.

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

    def forward(self, x):
        """Forward pass of channel attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Channel-attended tensor
        """
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp(avg_pool)
                logging.debug("Average pool attention shape: %s", str(channel_att_raw.shape))
            elif pool_type=='max':
                max_pool = F.max_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp(max_pool)
                logging.debug("Max pool attention shape: %s", str(channel_att_raw.shape))
            elif pool_type=='lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)
                logging.debug("LSE pool attention shape: %s", str(channel_att_raw.shape))

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).expand_as(x)
        logging.debug("Channel attention scale shape: %s", str(scale.shape))
        return x * scale

def logsumexp_2d(tensor):
    """Compute log-sum-exp pooling for 2D tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor

    Returns
    -------
    torch.Tensor
        Log-sum-exp pooled tensor
    """
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    """Channel pooling module that concatenates max and mean pooling results."""
    def forward(self, x):
        max_pool = torch.max(x,1)[0].unsqueeze(1)
        mean_pool = torch.mean(x,1).unsqueeze(1)
        logging.debug("Channel pool output shape: %s", str(torch.cat((max_pool, mean_pool), dim=1).shape))
        return torch.cat((max_pool, mean_pool), dim=1)

class SpatialGate(nn.Module):
    """Spatial attention module.

    Applies spatial attention using channel pooling and convolution.

    Parameters
    ----------
    kernel_size : int, optional
        Size of convolution kernel (default: 7)
    """
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, 
                               padding=(kernel_size-1) // 2, relu=False)
    
    def forward(self, x):
        """Forward pass of spatial attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Spatially-attended tensor
        """
        x_compress = self.compress(x)
        logging.debug("Spatial compress shape: %s", str(x_compress.shape))
        
        x_out = self.spatial(x_compress)
        logging.debug("Spatial conv shape: %s", str(x_out.shape))
        
        scale = F.sigmoid(x_out)
        logging.debug("Spatial attention scale shape: %s", str(scale.shape))
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

    Notes
    -----
    The module automatically handles both 3D (batch_size, channels, length)
    and 4D (batch_size, channels, 1, length) inputs.
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], 
                 no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.attention = None

    def forward(self, x):
        """Forward pass applying channel and optional spatial attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, length) or
            (batch_size, channels, 1, length)

        Returns
        -------
        torch.Tensor
            Attended tensor of same shape as input
        """
        logging.debug("CBAM input shape: %s", str(x.shape))
        
        x_out = self.ChannelGate(x)
        logging.debug("After channel attention shape: %s", str(x_out.shape))
        
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
            logging.debug("After spatial attention shape: %s", str(x_out.shape))

        self.attention = x_out / x
        logging.debug("Attention weights shape: %s", str(self.attention.shape))
        return x_out

    def get_attention(self):
        """Get the attention weights from the last forward pass.

        Returns
        -------
        torch.Tensor
            Attention weights tensor
        """
        return self.attention

# def get_cbam_attention(model, data_loader, device=torch.device("cuda")):
#     attention = []
    
#     model.eval()
#     for data, target in data_loader:
#         data, target = data.to(device), target.to(device)
#         pred = model(data)
#         batch_attention = model.Embedding.conv1.cbam.get_attention().cpu().data.numpy()
#         attention.append(batch_attention)

#     attention = np.concatenate(attention, 0)
#     return attention

# attention = get_cbam_attention(model, test_loader, device)
# filter_attention = attention.mean(0).mean(-1)

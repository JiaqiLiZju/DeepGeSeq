"""DGS Modules Package.

This package provides fundamental building blocks and modules for deep genomic sequence analysis:

Components
----------
BasicModule
    Base classes and common modules:
    - BasicModule: Abstract base class with weight initialization and model I/O
    - BasicConv1d: 1D convolution with batch norm and activation
    - BasicRNNModule: RNN/LSTM module with batch-first processing
    - BasicLinearModule: Linear layer with batch norm and activation
    - BasicPredictor: Task-specific prediction head
    - BasicLoss: Task-specific loss functions

SeqEmbed
    Sequence embedding modules:
    - BasicConvEmbed: Basic convolutional embedding
    - RevCompConvEmbed: Reverse complement aware embedding
    - CharConvModule: Character-level convolution

CBAM
    Attention modules:
    - CBAM: Convolutional Block Attention Module
    - ChannelGate: Channel attention mechanism
    - SpatialGate: Spatial attention mechanism

Notes
-----
All modules follow a consistent interface and support logging for debugging.
Modules are designed to work with one-hot encoded DNA sequences of shape
(batch_size, 4, sequence_length).
"""

from .BasicModule import *
from .SeqEmbed import *
from .CBAM import *

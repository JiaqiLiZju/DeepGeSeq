"""DGS Model Module.

This module provides various model architectures for deep genomic sequence analysis:

1. Basic Models:
   - BasicModel: Base class for all models
   - CNN: Basic Convolutional Neural Network
   - CAN: Convolutional Attention Network

2. Residual Models:
   - BasicBlock: Basic residual block
   - Bottleneck: Bottleneck residual block
   - ResNet: Residual Network architecture
   - ResidualNet: Factory function for creating ResNet models

3. Published Models:
   - DeepSEA: DeepSEA architecture (Zhou & Troyanskaya, 2015)
   - Beluga: DeepSEA architecture used in Expecto (Zhou & Troyanskaya, 2019)
   - DanQ: DanQ architecture (Quang & Xie, 2016)
   - Basset: Basset architecture (Kelley, 2016)
   - BPNet: BPNet architecture (Schreiber, 2018)
   - scBasset: scBasset architecture (Yuan et al, 2022)

4. Extension Utilities:
   - Module loading and instantiation utilities
"""

from .BasicModel import *
from .ConvModel import *
from .Publications import *
from .Extention import *
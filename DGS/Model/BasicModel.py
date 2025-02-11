"""Basic Model in DGS.
This module provides the foundational model architecture for deep genomic sequence analysis.

Classes
-------
BasicModel
    The general abstract base class for all models in DGS.
    Provides a standard architecture with Embedding, Encoder, Decoder, and Predictor components.
"""

# Code: jiaqili@zju.edu

import logging

import torch
from torch import nn

# TODO maybe not suitable for probability models
class BasicModel(nn.Module):
    """Basic Model class in DGS.
    
    A prototype for sequence-based deep-learning models. The architecture consists
    of four main components arranged in a sequential manner:

    1. Embedding: Transforms raw sequence input into vector representations
    2. Encoder: Processes embedded vectors to extract features
    3. Decoder: Transforms encoded features into a higher-level representation
    4. Predictor: Maps decoded features to task-specific outputs

    The model expects input sequences in the shape (batch_size, 4, sequence_length)
    where 4 represents the one-hot encoded nucleotides (A,C,G,T).

    Attributes
    ----------
    Embedding : nn.Sequential
        Layer(s) for embedding sequence input into vectors
    Encoder : nn.Sequential
        Layer(s) for encoding embedded vectors into feature maps
    Decoder : nn.Sequential
        Layer(s) for decoding feature maps into higher-level features
    Predictor : nn.Sequential
        Layer(s) for making final predictions

    Notes
    -----
    - Input sequences should be one-hot encoded with shape (batch_size, 4, sequence_length)
    - If input has shape (batch_size, sequence_length, 4), it will be automatically transposed
    - The Encoder output is flattened if it has more than 2 dimensions
    - Debug-level logging is available for shape tracking through the network

    Examples
    --------
    >>> model = BasicModel()
    >>> x = torch.randn(32, 4, 1000)  # batch of 32 sequences of length 1000
    >>> output = model(x)
    """
    def __init__(self):
        super().__init__()
        self.Embedding = nn.Sequential()
        self.Encoder = nn.Sequential()
        self.Decoder = nn.Sequential()
        self.Predictor = nn.Sequential()

    def forward(self, x):
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 4, sequence_length) or 
            (batch_size, sequence_length, 4)

        Returns
        -------
        torch.Tensor
            Model predictions with shape determined by the Predictor layer

        Notes
        -----
        The forward pass follows these steps:
        1. Transpose input if needed to (batch_size, 4, sequence_length)
        2. Pass through Embedding layer
        3. Pass through Encoder layer and flatten if needed
        4. Pass through Decoder layer
        5. Pass through Predictor layer to get final output
        """
        if x.shape[1] != 4:
            x = x.swapaxes(1, 2)

        embed = self.Embedding(x)
        logging.debug("Embedding output shape: %s", str(embed.shape))

        fmap = self.Encoder(embed)
        logging.debug("Encoder output shape: %s", str(fmap.shape))

        if len(fmap.shape) > 2:
            fmap = fmap.reshape((fmap.size(0), -1))
            logging.warning("Encoder output reshaped to (batch_size, -1). "
                          "Consider adding Flatten module to Encoder to avoid this warning.")

        fmap = self.Decoder(fmap)
        logging.debug("Decoder output shape: %s", str(fmap.shape))

        out = self.Predictor(fmap)
        logging.debug("Predictor output shape: %s", str(out.shape))

        return out


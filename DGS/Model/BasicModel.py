"""Basic Model in DGS.
This module provides 

1.  `BasicModel` class - the general abstract class

and supporting methods.
"""

# Code: jiaqili@zju.edu

import logging

import torch
from torch import nn

# TODO maybe not suitable for probability models
class BasicModel(nn.Module):
    """Basic Model class in DGS.
    Prototype for a sequence-based deep-learning model. 
    BasicModel contains Embedding, Encoder, Decoder, Predictor layers.
    
    Embedding : embed the sequence into vectors

    Encoder : encode the inputs into feature-maps

    Decoder : decode the encoded inputs (Flattened) into higher feature-maps

    Predictor : mapp the decoded feature-maps into task-specific space 
    and make prediction

    Tensor flows
    ------------
    -> Embedding(x)

    -> Encoder(x)

    -> Flatten(x)

    -> Decoder(x)

    -> Predictor(x)

    """
    def __init__(self):
        super().__init__()
        self.Embedding = nn.Sequential()
        self.Encoder = nn.Sequential()
        self.Decoder = nn.Sequential()
        self.Predictor = nn.Sequential()

    def forward(self, x):
        if x.shape[1] != 4:
            x = x.swapaxes(1, 2)

        embed = self.Embedding(x)
        logging.debug(embed.shape)

        fmap = self.Encoder(embed)
        logging.debug(fmap.shape)

        if len(fmap.shape) > 2:
            fmap = fmap.reshape((fmap.size(0), -1))
            logging.warning("fmap after Encoder reshaped as (batchsize, -1), \n \
                            Add Flatten module in Encoder to deprecate this warning")

        fmap = self.Decoder(fmap)
        logging.debug(fmap.shape)

        out = self.Predictor(fmap)
        logging.debug(out.shape)

        return out


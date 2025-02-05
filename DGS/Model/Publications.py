""" Reimplement of Publicated Model Archiectures applicable in DGS.
This module provides 

1. `DeepSEA` - DeepSEA architecture (Zhou & Troyanskaya, 2015).

2. `Beluga` - DeepSEA architecture used in Expecto (Zhou & Troyanskaya, 2019).

3. `DanQ` - DanQ architecture (Quang & Xie, 2016).

4. `Basset` - Basset architecture (Kelley, 2016).

5. `BPNet` - BPNet architecture (Schreiber, 2018).

6. `scBasset` - scBasset architecture (Yuan et al, 2022).

and supporting methods.
"""

import logging
import numpy as np

import torch
import torch.nn as nn

__all__ = ["DeepSEA", "Beluga", "DanQ", "Basset", "BPNet", "scBasset"]

class DeepSEA(nn.Module):
    """
    DeepSEA architecture (Zhou & Troyanskaya, 2015).
    """
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
        n_genomic_features : int
        """
        super(DeepSEA, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))

        reduce_by = conv_kernel_size - 1
        pool_kernel_size = float(pool_kernel_size)
        self.n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        self.classifier = nn.Sequential(
            nn.Linear(960 * self.n_channels, n_genomic_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid())

    def forward(self, x):
        """Forward propagation of a batch.
        """
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 960 * self.n_channels)
        predict = self.classifier(reshape_out)
        return predict

    def criterion(self):
        """
        The criterion the model aims to minimize.
        """
        return nn.BCELoss()

    def get_optimizer(self, lr):
        """
        The optimizer and the parameters with which to initialize the optimizer.
        At a later time, we initialize the optimizer by also passing in the model
        parameters (`model.parameters()`). We cannot initialize the optimizer
        until the model has been initialized.
        """
        return (torch.optim.SGD,
                {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class Beluga(nn.Module):
    """
    DeepSEA architecture used in Expecto (Zhou & Troyanskaya, 2019).
    """
    def __init__(self, sequence_length, n_genomic_features):
        super(Beluga, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 8
        n_hiddens = 32

        reduce_by = (conv_kernel_size - 1) * 2 # conv twice
        self.n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)

        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4,320,(1, conv_kernel_size)),
                nn.ReLU(),
                nn.Conv2d(320,320,(1, conv_kernel_size)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, pool_kernel_size),(1, pool_kernel_size)),
                nn.Conv2d(320,480,(1, conv_kernel_size)),
                nn.ReLU(),
                nn.Conv2d(480,480,(1, conv_kernel_size)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, pool_kernel_size),(1, pool_kernel_size)),
                nn.Conv2d(480,640,(1, conv_kernel_size)),
                nn.ReLU(),
                nn.Conv2d(640,640,(1, conv_kernel_size)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(640 * self.n_channels, n_hiddens)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(n_hiddens, n_genomic_features)),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(2) # update 2D sequences
        return self.model(x)


class DanQ(nn.Module):
    """
    DanQ architecture (Quang & Xie, 2016).
    """
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
            Input sequence length
        n_genomic_features : int
            Total number of features to predict
        """
        super(DanQ, self).__init__()
        self.nnet = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=26),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=13, stride=13),
            nn.Dropout(0.2))

        self.bdlstm = nn.Sequential(
            nn.LSTM(
                320, 320, num_layers=1, batch_first=True, bidirectional=True))

        self._n_channels = np.floor(
            (sequence_length - 25) / 13)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._n_channels * 640, 925),
            nn.ReLU(inplace=True),
            nn.Linear(925, n_genomic_features),
            nn.Sigmoid())

    def forward(self, x):
        """Forward propagation of a batch.
        """
        out = self.nnet(x)
        reshape_out = out.transpose(0, 1).transpose(0, 2)
        out, _ = self.bdlstm(reshape_out)
        out = out.transpose(0, 1)
        reshape_out = out.contiguous().view(
            out.size(0), 640 * self._n_channels)
        predict = self.classifier(reshape_out)
        return predict

    def criterion(self):
        return nn.BCELoss()

    def get_optimizer(self, lr):
        return (torch.optim.RMSprop, {"lr": lr})


class Basset(nn.Module):
    '''Basset architecture (Kelley, 2016).
    Deep convolutional neural networks for DNA sequence analysis.
    The architecture and optimization parameters for the DNaseI-seq compendium analyzed in the paper.
    '''
    def __init__(self, sequence_length, n_genomic_features):
        super(Basset, self).__init__()

        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4,300,(1, 19)),
                nn.BatchNorm2d(300),
                nn.ReLU(),
                nn.MaxPool2d((1, 8),(1, 8)),
                nn.Conv2d(300,200,(1, 11)),
                nn.BatchNorm2d(200),
                nn.ReLU(),
                nn.MaxPool2d((1, 8),(1, 8)),
                nn.Conv2d(200,200,(1, 7)),
                nn.BatchNorm2d(200),
                nn.ReLU(),
                nn.MaxPool2d((1, 8),(1, 8)),
            ),
            nn.Sequential(
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4800, 1000)),
                nn.BatchNorm1d(1000),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(1000, 32),
                nn.BatchNorm1d(32),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(32, n_genomic_features),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(2) # update 2D sequences
        return self.model(x)
    
    def architecture(self):
        d = {'conv_filters1':300,
            'conv_filters2':200,
            'conv_filters3':200,
            'conv_filter_sizes1':19,
            'conv_filter_sizes2':11,
            'conv_filter_sizes3':7,
            'pool_width1':3,
            'pool_width2':4,
            'pool_width3':4,
            'hidden_units1':1000,
            'hidden_units2':32,
            'hidden_dropouts1':0.3,
            'hidden_dropouts2':0.3,
            'learning_rate':0.002,
            'weight_norm':7,
            'momentum':0.98}
        return d


class BPNet(nn.Module):
    """
    This nn.Module was taken without permission from a Mr. Schreiber.
    Just kidding, he made it open source so, ipso facto, I do have permission.
    Anyway the documentation below is from him, so yell at him if it doesn't
    work.

    A basic BPNet model with stranded profile and total count prediction.
    This is a reference implementation for BPNet. The model takes in
    one-hot encoded sequence, runs it through:
    (1) a single wide convolution operation
    THEN
    (2) a user-defined number of dilated residual convolutions
    THEN
    (3a) profile predictions done using a very wide convolution layer
    that also takes in stranded control tracks
    AND
    (3b) total count prediction done using an average pooling on the output
    from 2 followed by concatenation with the log1p of the sum of the
    stranded control tracks and then run through a dense layer.
    This implementation differs from the original BPNet implementation in
    two ways:
    (1) The model concatenates stranded control tracks for profile
    prediction as opposed to adding the two strands together and also then
    smoothing that track
    (2) The control input for the count prediction task is the log1p of
    the strand-wise sum of the control tracks, as opposed to the raw
    counts themselves.
    (3) A single log softmax is applied across both strands such that
    the logsumexp of both strands together is 0. Put another way, the
    two strands are concatenated together, a log softmax is applied,
    and the MNLL loss is calculated on the concatenation.
    (4) The count prediction task is predicting the total counts across
    both strands. The counts are then distributed across strands according
    to the single log softmax from 3.
    
    Parameters
    ----------
    n_filters: int, optional
            The number of filters to use per convolution. Default is 64.
    n_layers: int, optional
            The number of dilated residual layers to include in the model.
            Default is 8.
    n_outputs: int, optional
            The number of profile outputs from the model. Generally either 1 or 2
            depending on if the data is unstranded or stranded. Default is 2.
    alpha: float, optional
            The weight to put on the count loss.
    name: str or None, optional
            The name to save the model to during training.
    trimming: int or None, optional
            The amount to trim from both sides of the input window to get the
            output window. This value is removed from both sides, so the total
            number of positions removed is 2*trimming.
    verbose: bool, optional
            Whether to display statistics during training. Setting this to False
            will still save the file at the end, but does not print anything to
            screen during training. Default is True.
    """

    def __init__(
        self,
        input_len,
        output_dim,
        n_filters=64,
        n_layers=8,
        n_outputs=2,
        n_control_tracks=2,
        alpha=1,
        profile_output_bias=True,
        count_output_bias=True,
        name=None,
        trimming=None,
        verbose=True,
    ):
        super(BPNet, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.output_dim = output_dim
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.n_control_tracks = n_control_tracks
        self.alpha = alpha
        self.name = name or "bpnet.{}.{}".format(n_filters, n_layers)
        self.trimming = trimming or 2**n_layers

        # Build the model
        self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
        self.irelu = torch.nn.ReLU()

        self.rconvs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    n_filters, n_filters, kernel_size=3, padding=2**i, dilation=2**i
                )
                for i in range(1, self.n_layers + 1)
            ]
        )
        self.rrelus = torch.nn.ModuleList(
            [torch.nn.ReLU() for i in range(1, self.n_layers + 1)]
        )

        self.fconv = torch.nn.Conv1d(
            n_filters + n_control_tracks,
            n_outputs,
            kernel_size=75,
            padding=37,
            bias=profile_output_bias,
        )

        n_count_control = 1 if n_control_tracks > 0 else 0
        self.linear = torch.nn.Linear(
            n_filters + n_count_control, 1, bias=count_output_bias
        )

    def forward(self, X, X_ctl=None):
        """A forward pass of the model.
        
        This method takes in a nucleotide sequence X, a corresponding
        per-position value from a control track, and a per-locus value
        from the control track and makes predictions for the profile
        and for the counts. This per-locus value is usually the
        log(sum(X_ctl_profile)+1) when the control is an experimental
        read track but can also be the output from another model.
        
        Parameters
        ----------
        X: torch.tensor, shape=(batch_size, 4, sequence_length)
                The one-hot encoded batch of sequences.
        X_ctl: torch.tensor, shape=(batch_size, n_strands, sequence_length)
                A value representing the signal of the control at each position in
                the sequence.
       
        Returns
        -------
        y_profile: torch.tensor, shape=(batch_size, n_strands, out_length)
                The output predictions for each strand.
        """
        start, end = self.trimming, X.shape[2] - self.trimming

        X = self.irelu(self.iconv(X))
        for i in range(self.n_layers):
            X_conv = self.rrelus[i](self.rconvs[i](X))
            X = torch.add(X, X_conv)

        if X_ctl is None:
            X_w_ctl = X
        else:
            X_w_ctl = torch.cat([X, X_ctl], dim=1)

        y_profile = self.fconv(X_w_ctl)[:, :, start:end]

        # counts prediction
        X = torch.mean(X[:, :, start - 37 : end + 37], dim=2)

        if X_ctl is not None:
            X_ctl = torch.sum(X_ctl[:, :, start - 37 : end + 37], dim=(1, 2))
            X_ctl = X_ctl.unsqueeze(-1)
            X = torch.cat([X, torch.log(X_ctl + 1)], dim=-1)

        y_counts = self.linear(X).reshape(X.shape[0], 1)
        return y_profile, y_counts

class scBasset(nn.Module):
    """scBasset model implementation from Yuan et al 2022 in PyTorch

    WARNING: This model has not been fully tested yet. Use at your own risk.

    Parameters
    ----------
    num_cells : int
        The number of cells in the dataset.
    num_batches : int
        The number of batches in the dataset. If not specified, the model will
        not include batch correction.
    l1 : float
        The L1 regularization parameter for the cell layer.
    l2 : float
        The L2 regularization parameter for the batch layer.
    """
    def __init__(self, num_cells, num_batches=None):
        super(scBasset, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=288, kernel_size=17, padding=8),
            nn.BatchNorm1d(288),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3),
        )

        self.conv_tower = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(in_channels=288, out_channels=288, kernel_size=5, padding=2),
                nn.BatchNorm1d(288),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=288, out_channels=323, kernel_size=5, padding=2),
                nn.BatchNorm1d(323),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=323, out_channels=363, kernel_size=5, padding=2),
                nn.BatchNorm1d(363),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=363, out_channels=407, kernel_size=5, padding=2),
                nn.BatchNorm1d(407),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=407, out_channels=456, kernel_size=5, padding=2),
                nn.BatchNorm1d(456),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=456, out_channels=512, kernel_size=5, padding=2),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )

        self.flatten = nn.Flatten()

        self.bottleneck = nn.Sequential(
            nn.Linear(in_features=1792, out_features=32),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.2),
            nn.GELU(),
        )

        self.fc1 = nn.Linear(in_features=32, out_features=num_cells)
        self.sigmoid = nn.Sigmoid()

        if num_batches is not None:
            self.fc2 = nn.Linear(in_features=32, out_features=num_batches)

    def forward(self, x, batch=None):
        x = self.conv1(x)
        x = self.conv_tower(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.bottleneck(x)
        if batch is not None:
            x_batch = self.fc2(x)
            x_batch = torch.matmul(x_batch, batch)
            x = self.fc1(x) + x_batch
        else:
            x = self.fc1(x)

        x = self.sigmoid(x)

        return x

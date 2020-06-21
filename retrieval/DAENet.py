
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from torch.autograd import Function
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import Function
import numpy as np
import pdb


###################################
########  densenet blocks #########
###################################

class DenseBlockEncoder(nn.Module):
    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False], norm_layer=nn.BatchNorm2d):
        super(DenseBlockEncoder, self).__init__()
        assert(n_convs > 0)

        self.n_channels = n_channels
        self.n_convs    = n_convs
        self.layers     = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(
                    norm_layer(n_channels),
                    activation(*args),
                    nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False),))

    def forward(self, inputs):
        outputs = []

        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]


class DenseBlockDecoder(nn.Module):
    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False], norm_layer=nn.BatchNorm2d):
        super(DenseBlockDecoder, self).__init__()
        assert(n_convs > 0)

        self.n_channels = n_channels
        self.n_convs    = n_convs
        self.layers = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(
                    norm_layer(n_channels),
                    activation(*args),
                    nn.ConvTranspose2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False),))

    def forward(self, inputs):
        outputs = []

        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]


class DenseTransitionBlockEncoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, mp, activation=nn.ReLU, args=[False], norm_layer=nn.BatchNorm2d):
        super(DenseTransitionBlockEncoder, self).__init__()
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.mp             = mp
        self.main           = nn.Sequential(
                norm_layer(n_channels_in),
                activation(*args),
                nn.Conv2d(n_channels_in, n_channels_out, 1, stride=1, padding=0, bias=False),
                nn.MaxPool2d(mp),
        )
    def forward(self, inputs):
        return self.main(inputs)


class DenseTransitionBlockDecoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, activation=nn.ReLU, args=[False], norm_layer=nn.BatchNorm2d):
        super(DenseTransitionBlockDecoder, self).__init__()
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.main           = nn.Sequential(
                norm_layer(n_channels_in),
                activation(*args),
                nn.ConvTranspose2d(n_channels_in, n_channels_out, 4, stride=2, padding=1, bias=False),
        )
    def forward(self, inputs):
        return self.main(inputs)



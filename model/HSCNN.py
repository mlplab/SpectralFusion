# coding: utf-8


import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from colour.colorimetry import transformations
import torch
import torchvision
from torchinfo import summary
from .layers import ReLU, Leaky, Swish, Mish


class HSCNN(torch.nn.Module):

    '''
    HSCNN

    Attributes:
    '''
    def __init__(self, input_ch: int, output_ch: int, *args, feature: int=64,
                 block_num: int=9, **kwargs) -> None:
        super(HSCNN, self).__init__()
        activation = kwargs.get('activation', 'leaky')
        activations = {'relu': ReLU, 'leaky': Leaky, 'swish': Swish, 'mish': Mish}
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1)
        self.start_activation = activations[activation]()
        self.patch_extraction = torch.nn.Conv2d(output_ch, feature, 3, 1, 1)
        self.feature_map = torch.nn.ModuleDict({f'Conv_{i}': 
            torch.nn.Conv2d(feature, feature, 3, 1, 1) for i in range(block_num - 1)})
        self.activations = torch.nn.ModuleDict({
            f'Conv_{i}': activations[activation]() for i in range(block_num - 1)})
        self.residual_conv = torch.nn.Conv2d(feature, output_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.start_conv(x)
        x_in = x
        x = self.start_activation(self.patch_extraction(x))
        for (layer), (activation) in zip(self.feature_map.values(), self.activations.values()):
            x = activation(layer(x))
        output = self.residual_conv(x) + x_in
        return output


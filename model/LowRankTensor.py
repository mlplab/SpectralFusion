# coding: utf-8


import torch
import torchvision
from layers import Base_Module


class GAP(Base_Module):

    def __init__(self, *args, dims=(2, 3), **kwargs):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.mean(x, dim=self.dims)
        return x


class CPdecomposition(Base_Module):

    def __init__(self, input_ch, output_ch, *args, feature_num: int=64, **kwargs) -> None:
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

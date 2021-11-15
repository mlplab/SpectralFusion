# coding: UTF-8


import torch
from torchinfo import summary
from .layers import Base_Module, Attention_HSI_Block


class SSAttention(Base_Module):

    def __init__(self, input_ch: int, output_ch: int, *args, feature_num: int=64,
                 layer_num: int=9, **kwargs) -> None:
        super().__init__()
        ratio = kwargs.get('ratio', 4)
        self.input_conv = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1)
        self.attention_layers = torch.nn.ModuleDict({f'Attn_{i}': Attention_HSI_Block(output_ch, output_ch, feature_num=feature_num, ratio=ratio) for i in range(layer_num)})
        self.residual_layers = torch.nn.ModuleDict({f'Res_{i}': torch.nn.Conv2d(output_ch, output_ch, 3, 1, 1) for i in range(layer_num)})
        self.output_conv = torch.nn.Conv2d(output_ch, output_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input_conv(x)
        x_in = x
        for attn_layer, res_layer in zip(self.attention_layers.values(), 
                                         self.residual_layers.values()):
            attn_x = attn_layer(x)
            res_x = res_layer(x)
            x = x_in + attn_x + res_x
        x = self.output_conv(x)
        return x

# coding: UTF-8


import torch
from torchinfo import summary
from .layers import Base_Module, Attention_HSI_Block


class SSAttention(Base_Module):

    '''
    Attentiont 機構を用いた再構成モデル

    Attributes:
        input_conv (torch.nn.Module): 入力時の畳み込み処理. 
        attention_layers (torch.nn.Module): Attention 機構を用いた中間処理層
        residual_layers (torch.nn.Module): residual 構造を用いた中間処理層
        output_conv (torch.nn.Module): 出力時の畳み込み層
        '''

    def __init__(self, input_ch: int, output_ch: int, *args, feature_num: int=64,
                 layer_num: int=9, **kwargs) -> None:
        '''
        Args:
            input_ch (int):
                入力画像のチャンネル数. 
            output_ch (int):
                出力画像のチャンネル数. 
            feature_num (int, optional):
                中間層のチャンネル数. Default = 64
            layer_num (int, optional):
                中間層の数.  Default = 9
            ratio (int, optional):
                Spectral Attention の圧縮率. Default = 4
            '''
        super().__init__()
        ratio = kwargs.get('ratio', 4)
        self.input_conv = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1)
        self.attention_layers = torch.nn.ModuleDict(
                {f'Attn_{i}': Attention_HSI_Block(
                    output_ch, output_ch, feature_num=feature_num, ratio=ratio) 
                    for i in range(layer_num)
                    }
                )
        self.residual_layers = torch.nn.ModuleDict(
                {f'Res_{i}': torch.nn.Conv2d(
                    output_ch, output_ch, 3, 1, 1) for i in range(layer_num)
                    }
                )
        self.output_conv = torch.nn.Conv2d(output_ch, output_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        実際の再構成処理

        Args:
            x (torch.Tensor): 入力画像
        Returns:
            output (torch.Tensor): 出力画像
        '''

        x = self.input_conv(x)
        x_in = x
        for attn_layer, res_layer in zip(self.attention_layers.values(), 
                                         self.residual_layers.values()):
            attn_x = attn_layer(x)
            res_x = res_layer(x)
            x = x_in + attn_x + res_x
        output = self.output_conv(x)
        return output

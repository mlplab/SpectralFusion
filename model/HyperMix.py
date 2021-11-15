# coding: UTF-8


import torch
from .layers import Base_Module, Mix_Conv, Mix_SS_Layer


class Mix_Reconst_Net(Base_Module):

    def __init__(self, input_ch: int, output_ch: int, *args, chunks: int=2,
            block_num: int=9, feature_num: int=64, **kwargs) -> None:
        super(Mix_Reconst_Net, self).__init__()

        activation = kwargs.get('activation', 'ReLU').lower()
        group_num = kwargs.get('group_num', 1)
        se_flag = kwargs.get('se_flag')
        ratio = kwargs.get('ratio', 2)
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1)
        self.mix_ss_layers = torch.nn.ModuleList([Mix_SS_Layer(output_ch,
                                                               output_ch, chunks,
                                                               feature_num=feature_num,
                                                               group_num=group_num,
                                                               activation=activation,
                                                               se_flag=se_flag,
                                                               ratio=ratio) for _ in range(block_num)])
        self.output_conv = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.start_conv(x)
        for mix_ss_layer in self.mix_ss_layers:
            x = mix_ss_layer(x)
        return self.output_conv(x)

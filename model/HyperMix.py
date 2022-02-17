# coding: UTF-8


import torch
from .layers import Base_Module, Mix_Conv, Mix_SS_Layer


class Mix_Reconst_Net(Base_Module):

    '''
    HyperMixNet

    Attributes:
        input_conv (torch.nn.Module): 入力時の畳み込み層. スペクトル成分の拡張を行う．
        mix_ss_layers (torch.nn.ModuleDict[torch.nn.Module]): 中間層.
        output_conv (torch.nn.Module): 出力時の畳み込み層. 
    '''
        

    def __init__(self, input_ch: int, output_ch: int, *args, chunks: int=2,
            layer_num: int=9, feature_num: int=64, **kwargs) -> None:
        '''
        Parameters:
            input_ch (int): 入力チャンネル数.
            output_ch (int): 出力チャンネル数.
            chunks (int, optional): MixConv の分割数. Default = 2
            layer_num (int, optional): モデルの中間数. Default = 9
            feature_num (int, optional): 中間層のチャンネル数. Default = 64
            activation (str, optional):
                中間層に使用する活性化関数.  ReLU('relu'), Leaky ReLU('leaky'), 
                Swish('swish'), Mish('mish') から選択可能. 
                Default = 'relu'
        '''
        super(Mix_Reconst_Net, self).__init__()
        activation = kwargs.get('activation', 'ReLU').lower()
        self.input_conv = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1)
        self.mix_ss_layers = torch.nn.ModuleList([Mix_SS_Layer(output_ch,
                                                               output_ch, chunks,
                                                               feature_num=feature_num,
                                                               activation=activation,
                                                               ratio=ratio) 
                                                               for _ in range(layer_num)])
        self.output_conv = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        '''
        実際の再構成を行う

        Args:
            x (torch.Tensor): 入力画像
        Returns:
            output (torch.Tneosr): 出力画像
        '''
            
        x = self.input_conv(x)
        for mix_ss_layer in self.mix_ss_layers:
            x = mix_ss_layer(x)
        output = self.output_conv(x)
        return output

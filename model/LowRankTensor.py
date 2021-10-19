# coding: utf-8


import torch
import torchvision
from .layers import Base_Module


class GAP(Base_Module):

    def __init__(self, *args, dims=(2, 3), **kwargs):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.mean(x, dim=self.dims, keepdims=True)
        return x


class CPdecomposition(Base_Module):

    def __init__(self, input_ch, output_ch, *args, h_kernel: int=1,
            w_kernel: int=1, feature_num: int=64, **kwargs) -> None:
        super().__init__()
        self.dims = ['h', 'w', 'c']
        gap_dims = {'h': (1, 3), 'w': (1, 2), 'c': (2, 3)}
        self.gap = torch.nn.ModuleDict({f'GAP_{dim}': GAP(dims=gap_dims[dim])
                                        for dim in self.dims})
        self.sig = torch.nn.ModuleDict({f'Sig_{dim}': torch.nn.Sigmoid()
                                        for dim in self.dims})
        self.conv = torch.nn.ModuleDict({'Conv_h': torch.nn.Conv2d(1, 1, (h_kernel, 1), 1, (h_kernel // 2, 0)),
                                         'Conv_w': torch.nn.Conv2d(1, 1, (1, w_kernel), 1, (0, w_kernel // 2)),
                                         'Conv_c': torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        h_feature = self.sig['Sig_h'](self.conv['Conv_h'](self.gap['GAP_h'](x)))
        w_feature = self.sig['Sig_w'](self.conv['Conv_w'](self.gap['GAP_w'](x)))
        c_feature = self.sig['Sig_c'](self.conv['Conv_c'](self.gap['GAP_c'](x)))
        hw_feature = torch.matmul(h_feature, w_feature)
        hw_feature = hw_feature.reshape(hw_feature.shape[0], hw_feature.shape[1], -1)
        output_feature = torch.matmul(c_feature, hw_feature)
        output_feature = output_feature.reshape(x.shape)
        return output_feature


class CPResidual(Base_Module):

    def __init__(self, input_ch, output_ch, *args, h_kernel: int=1,
            w_kernel: int=1, feature_num: int=64, **kwargs) -> None:
        super().__init__()
        self.CPUp = CPdecomposition(input_ch, input_ch, h_kernel=h_kernel, w_kernel=w_kernel)
        self.CPDown = CPdecomposition(input_ch, input_ch, h_kernel=h_kernel, w_kernel=w_kernel)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        xup = self.CPUp(x)
        x = xup - x
        xdown = self.CPDown(x)
        xdown = xdown + x
        return xup, xdown


class RankFusion(Base_Module):

    def __init__(self, input_ch, output_ch, *args, feature_num: int=64, rank: int=3, **kwargs) -> None:
        super().__init__()
        self.concatConv = torch.nn.Conv2d(input_ch * rank, output_ch, 1, 1, 0)

    def forward(self, x: torch.Tensor, low_rank: torch.Tensor) -> torch.Tensor:
        low_rank = self.concatConv(low_rank)
        output = low_rank * x
        return output


class EncodingConv(Base_Module):

    def __init__(self, input_ch, output_ch, *args, feature_num: int=64, **kwargs) -> None:

        super().__init__()
        activation = kwargs.get('activation', 'relu')
        self.conv1 = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)
        self.activation = self.activations[activation]()
        self.conv2 = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return x


class DecodingConv(Base_Module):

    def __init__(self, input_ch, output_ch, *args, feature_num: int=64, **kwargs) -> None:

        super().__init__()
        activation = kwargs.get('activation', 'relu')
        self.conv1 = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1)
        self.activation = self.activations[activation]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x))
        return x


class CPModule(Base_Module):

    def __init__(self, input_ch, output_ch, *args, h_kernel: int=1,
                 w_kernel: int=1, feature_num: int=64, rank: int=3, 
                 data_type: str='None', **kwargs) -> None:
        super().__init__()
        self.rank_num = rank
        activation = kwargs.get('activation', 'relu')
        self.data_type = data_type
        self.input_conv = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)
        self.encoding = EncodingConv(feature_num, feature_num)
        self.init_CPlayer = CPResidual(feature_num, feature_num, h_kernel, w_kernel)
        self.CP_layer = torch.nn.ModuleDict({f'{data_type}_CP_{i}': CPResidual(feature_num, feature_num, h_kernel, w_kernel)
                                             for i in range(1, rank)})
        self.low_rank_fusion = RankFusion(feature_num, feature_num, rank=rank)
        self.decoding = DecodingConv(feature_num, output_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input_conv(x)
        x_enc = self.encoding(x)
        xup, xdown = self.init_CPlayer(x_enc)
        tmp_xup = xdown
        output = xup
        for rank in range(1, self.rank_num):
            tmp_xup, tmp_xdown = self.CP_layer[f'{self.data_type}_CP_{rank}'](tmp_xup)
            xup = xup + tmp_xup
            output = torch.cat([output, xup], dim=1)
            tmp_xup = tmp_xdown
        output = self.low_rank_fusion(x_enc, output)
        output = output + x
        output = self.decoding(output)
        return output


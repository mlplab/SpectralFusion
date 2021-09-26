# coding: utf-8


import torch
from  torchinfo import summary
from .layers import DW_PT_Conv


class ConvEmbed(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, *args,
                 feature_num: int=64, kernel_size: int=3, **kwargs) -> None:
        super().__init__()
        self.conv_layer = torch.nn.Conv2d(input_ch, output_ch, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.conv_layer(x)
        return x


class ConvProj(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, *args,
                 feature_num: int=64, kernel_size: int=3, stride: int=1, **kwargs) -> None:
        super().__init__()
        self.conv_layer = DW_PT_Conv(input_ch, output_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.conv_layer(x)
        x = x.view(b, -1, c)
        return x


class MultiHeadConv(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, head_num: int, *args,
                 feature_num: int=64, kernel_size: int=3, **kwargs) -> None:
        super().__init__()

        self.head_num = head_num
        self.q_conv = ConvProj(input_ch, output_ch)
        self.k_conv = ConvProj(input_ch, output_ch, kernel_size=kernel_size, stride=2, padding=kernel_size // 2)
        self.v_conv = ConvProj(input_ch, output_ch, kernel_size=kernel_size, stride=2, padding=kernel_size // 2)
        # self.norm_aw = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor):

        b, c, h ,w = x.shape
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)
        q = torch.reshape(q.unsqueeze(1), (b, self.head_num, h * w, c // self.head_num))
        k = torch.reshape(k.unsqueeze(1), (b, self.head_num, h * w // 4, c // self.head_num))
        v = torch.reshape(v.unsqueeze(1), (b, self.head_num, h * w // 4, c // self.head_num))

        attention_weight = torch.matmul(q, k.permute(0, 1, 3, 2))
        attention_weight = torch.sigmoid(attention_weight)
<<<<<<< HEAD
=======
        print(v.shape)
        print(attention_weight.shape)
>>>>>>> 10483b3318a7979b01d3acec64dca2c35799bde7
        output = torch.matmul(attention_weight, v)
        output = torch.reshape(output, (b, c, h, w))
        return output


class OutputConv(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, *args, feature_num: int=64,
                 alpha: float=1., **kwargs) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(input_ch, int(output_ch * alpha), 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(int(output_ch * alpha), output_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv1(x1)
        return x2



class CvT(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, head_num: int, *args,
                 feature_num: int=64, layer_num=3, alpha: float=1., **kwargs) -> None:
        super().__init__()

        self.layer_num = layer_num
        self.start_conv = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)
        self.embedding = torch.nn.ModuleDict({f'Embed_{idx + 1}': ConvEmbed(feature_num, feature_num, feature_num=feature_num)
                                              for idx in range(layer_num)})
        self.attn = torch.nn.ModuleDict({f'Attn_{idx + 1}': MultiHeadConv(feature_num, feature_num, head_num, feature_num=feature_num)
                                         for idx in range(layer_num)})
        self.layer_out_conv = torch.nn.ModuleDict({f'Out_Conv_{idx + 1}': OutputConv(feature_num, feature_num, alpha=alpha)
                                                   for idx in range(layer_num)})
        self.output_conv = torch.nn.Conv2d(feature_num, output_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.start_conv(x)
        for idx in range(self.layer_num):
            x_in = x
            x = self.embedding[f'Embed_{idx + 1}'](x)
            x = self.attn[f'Attn_{idx + 1}'](x)
            x_in = x + x_in
            x = self.layer_out_conv[f'Out_Conv_{idx + 1}'](x)
            x = x + x_in
        x = self.output_conv(x)
        return x


if __name__ == '__main__':

    model = CvT(1, 31, 1)
    summary(model, (1, 1, 64, 64))

# coding: utf-8


import torch
from  torchinfo import summary
from layers import DW_PT_Conv


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
                 feature_num: int=64, kernel_size: int=1, patch_size: int=64,
                 h: int=512, w: int=512, **kwargs) -> None:
        super().__init__()

        self.head_num = head_num
        self.patch_size = patch_size
        self.h_max = h // patch_size
        self.w_max = w // patch_size
        self.q_conv = ConvProj(input_ch, output_ch, kernel_size=kernel_size)
        self.k_conv = ConvProj(input_ch, output_ch, kernel_size=kernel_size, stride=1)
        self.v_conv = ConvProj(input_ch, output_ch, kernel_size=kernel_size, stride=1)
        self.position_encode = torch.nn.ParameterDict({'H': torch.nn.Parameter(torch.rand(1, output_ch, h, 1)),
                                                       'W': torch.nn.Parameter(torch.rand(1, output_ch, 1, w))})
        # self.norm_aw = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, patch_idx: int) -> torch.Tensor:

        b, c, h ,w = x.shape
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)
        q = torch.reshape(q.unsqueeze(1), (b, self.head_num, h * w, c // self.head_num))
        k = torch.reshape(k.unsqueeze(1), (b, self.head_num, h * w, c // self.head_num))
        v = torch.reshape(v.unsqueeze(1), (b, self.head_num, h * w, c // self.head_num))

        w_idx = patch_idx % self.w_max
        r_w = self.position_encode['W'][:, :, :, self.patch_size * w_idx: self.patch_size * (w_idx + 1)]
        h_idx = patch_idx // self.h_max
        h_w = self.position_encode['H'][:, :, self.patch_size * h_idx: self.patch_size * (h_idx + 1), :]
        position_weight = h_w + r_w
        position_weight = torch.reshape(position_weight, (1, self.head_num, h * w, c // self.head_num))
        position_weight = torch.matmul(q, position_weight.permute(0, 1, 3, 2))

        attention_weight = torch.matmul(q, k.permute(0, 1, 3, 2))
        attention_weight = attention_weight + position_weight
        attention_weight = torch.softmax(attention_weight, dim=1)
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
        h = kwargs.get('h', 512)
        w = kwargs.get('w', 512)
        patch_size = kwargs.get('patch_size', 64)
        self.start_conv = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)
        self.conv_layer = torch.nn.ModuleDict({f'Conv_{idx + 1}': torch.nn.Conv2d(feature_num, feature_num, 3, 1, 1)
                                           for idx in range(layer_num)})
        self.activation_layer = torch.nn.ModuleDict({f'Act_{idx + 1}': torch.nn.ReLU()
                                                     for idx in range(layer_num)})
        self.embedding = torch.nn.ModuleDict({'Embed': ConvEmbed(feature_num, feature_num, feature_num=feature_num)})
        self.attn = torch.nn.ModuleDict({'Attn': MultiHeadConv(feature_num, feature_num, head_num, feature_num=feature_num, patch_size=patch_size,
                                                               h=h, w=w)})

        self.layer_out_conv = torch.nn.ModuleDict({'Out_Conv': OutputConv(feature_num, feature_num, alpha=alpha)})
        self.output_conv = torch.nn.Conv2d(feature_num, output_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor, patch_idx: int) -> torch.Tensor:

        x = self.start_conv(x)
        x_in = x
        for idx in range(self.layer_num):
            x = self.activation_layer[f'Act_{idx + 1}'](self.conv_layer[f'Conv_{idx + 1}'](x))
            x = x + x_in
            x_in = x
        x = self.embedding['Embed'](x)
        x = self.attn['Attn'](x, patch_idx)
        x_in = x + x_in
        x = self.layer_out_conv['Out_Conv'](x)
        x = x + x_in
        x = self.output_conv(x)
        return x


if __name__ == '__main__':

    model = CvT(1, 31, 1)
    # summary(model, ((1, 1, 64, 64), 1))
    input_x = torch.rand((1, 1, 64, 64))
    y = model(input_x, 1)
    print(y.shape)

# coding: UTF-8


import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torchvision
from torchinfo import summary
from .layers import Base_Module, Attention_HSI_Block

sns.set()


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def psnr(x, y):
    if len(x.shape) == 4:
        _, c, _, _ = x.shape
    elif len(x.shape) == 3:
        c, _, _ = x.shape
    else:
        print('shiran')
    all_psnr = 10. * torch.Tensor([torch.log10(y[:, i].max() ** 2. / ((x[:, i] - y[:, i]) ** 2)).mean() for i in range(c)])
    return torch.mean(all_psnr)


def sam(x, y):
    if len(x.shape) == 4:
        x_sqrt = torch.norm(x, dim=1)
        y_sqrt = torch.norm(y, dim=1)
        # print('sam')
        # print(x_sqrt.shape, y_sqrt.shape)
        xy = torch.sum(x * y, dim=1)
    elif len(x.shape) == 3:
        x_sqrt = torch.norm(x, dim=0)
        y_sqrt = torch.norm(y, dim=0)
        # print('sam')
        # print(x_sqrt.shape, y_sqrt.shape)
        xy = torch.sum(x * y, dim=0)
    metrics = xy / (x_sqrt * y_sqrt + 1e-6)
    angle = torch.acos(metrics)
    return torch.mean(angle)


class SSAttention(Base_Module):

    def __init__(self, input_ch: int, output_ch: int, *args, feature_num: int=64,
                 layer_num: int=9, **kwargs) -> None:
        super().__init__()
        ratio = kwargs.get('ratio', 4)
        self.output_ch = output_ch
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

    def get_attention_map(self, x: torch.Tensor, spatial_flag: bool=True):

        spatial_map = {}
        spectral_map = {}
        spatial_feature = {}
        x = self.input_conv(x)
        x_in = x
        for i, (attn_layer, res_layer) in enumerate(zip(self.attention_layers.values(),
                                         self.residual_layers.values())):
            attn_x, attn_map, h = attn_layer.get_attention_map(x ,spatial_flag=spatial_flag)
            spatial_map[f'Spatial_{i}'] = attn_map[0]
            spectral_map[f'Spectral_{i}'] = attn_map[1]
            spatial_feature[f'Spatial_Feature_{i}'] = h
            res_x = res_layer(x)
            x = x_in + attn_x + res_x
        x = self.output_conv(x)
        return x, spatial_map, spectral_map, spatial_feature

    def plot_attention_map(self, x: torch.Tensor, y: torch.Tensor, save_dir: str='Attention',
                           spatial_map: bool=True, *args, **kwargs):

        mat_mode = kwargs.get('mat_mode', False)
        row, col = int(np.ceil(np.sqrt(self.output_ch))), int(np.ceil(np.sqrt(self.output_ch)))
        os.makedirs(save_dir, exist_ok=True)
        output, spatial_map , spectral_map, spatial_feature = self.get_attention_map(x)
        os.makedirs(os.path.join(save_dir, 'spatial_map'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'spatial_colorbar_map'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'spectral_map'), exist_ok=True)

        print('spatial')
        for (layer_name, layer), (spatial_layer_name, spatial_layer) in zip(spatial_map.items(), spatial_feature.items()):
            # img = plt.imshow(plot_feature, cmap='jet')
            diff = normalize((layer - spatial_layer).abs()).permute(1, 0, 2, 3)
            plot_feature = torchvision.utils.make_grid(diff, nrow=row, padding=0).detach().cpu().numpy().copy()
            plot_feature = plot_feature[0] * .299 + plot_feature[1] * .587 + plot_feature[2] * .114
            plt.imsave(os.path.join(save_dir, 'spatial_map', f'{layer_name}_map.png'), plot_feature, cmap='jet')
            plt.figure(figsize=(8, 6))
            plt.imshow(plot_feature, cmap='jet')
            plt.axis('off')
            plt.colorbar()
            plt.savefig(os.path.join(save_dir, 'spatial_colorbar_map', f'{layer_name}_map.png'), bbox_inches='tight')
            plt.clf()
        print('spectral')
        for (layer_name, layer), (spatial_layer_name, spatial_layer) in zip(spectral_map.items(), spatial_feature.items()):
            plot_spatial_layer = normalize(spatial_layer).permute(1, 0, 2, 3)
            plot_feature = torchvision.utils.make_grid(plot_spatial_layer, nrow=row, padding=0).detach().cpu().numpy().copy()
            plot_feature = plot_feature[0] * .299 + plot_feature[1] * .587 + plot_feature[2] * .114
            plt.imsave(os.path.join(save_dir, 'spectral_map', f'{spatial_layer_name}.png'), plot_feature, cmap='jet')
            plot_spectral_score = np.ones((64, 64, 31))
            spectral_map = normalize(layer).squeeze(0).detach().cpu().numpy().copy().transpose(1, 2, 0)
            plot_spectral_score *= spectral_map
            plot_spectral_score = torch.Tensor(np.expand_dims(plot_spectral_score.transpose(2, 0, 1), axis=1))
            # plot_spectral_score = torch.ones_like(plot_spectral_score)
            plot_spectral_score = torchvision.utils.make_grid(plot_spectral_score, nrow=row, padding=0).detach().cpu().numpy().copy()[0]
            # print(plot_spectral_score[:, 0, 0], spectral_map[:, :, 0])
            plt.imsave(os.path.join(save_dir, 'spectral_map', f'{layer_name}_map.png'), plot_spectral_score, cmap='jet')

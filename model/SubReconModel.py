# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import torch
from colour.colorimetry import transformations
from .layers import Base_Module


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


class RGBHSCNN(Base_Module):

    def __init__(self, input_ch: int, output_ch: int, *args, feature_num: int=64,
                 layer_num: int=3, **kwargs) -> None:
        super().__init__()
        activation = kwargs.get('activation', 'relu').lower()
        self.output_ch = output_ch
        if input_ch == 0:
            self.input_conv = torch.nn.Identity()
        else:
            self.input_conv = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)
        self.input_activation = self.activations[activation]()
        self.feature_layers = torch.nn.ModuleDict({f'RGB_{i}': torch.nn.Conv2d(feature_num, feature_num, 3, 1, 1)
                                                   for i in range(layer_num)})
        self.activation_layer = torch.nn.ModuleDict({f'RGB_act_{i}': self.activations[activation]()
                                                     for i in range(layer_num)})
        self.output_conv = torch.nn.Conv2d(feature_num, output_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input_activation(self.input_conv(x))
        for layer, activation in zip(self.feature_layers.values(), self.activation_layer.values()):
            x = activation(layer(x))
        x = self.output_conv(x)
        return x

    def get_feature(self, x: torch.Tensor, pick_layer: list) -> dict:

        return_features = {}
        x = self.input_conv(x)
        x_in = x
        if 'start_conv' in pick_layer:
            return_features['start_conv'] = x_in
        x = self.input_activation(x)
        for (layer_name, layer), (activation_name, activation) in zip(self.feature_layers.items(), self.activation_layer.items()):
            x = activation(layer(x))
            if activation_name in pick_layer:
                return_features[activation_name] = x
        '''
        if self.residual:
            output = self.residual_conv(x) + x_in
        else:
            output = self.residual_conv(x)
        '''
        output = self.output_conv(x)
        if 'output_conv' in pick_layer:
            return_features['output_conv'] = output
        return return_features

    def plot_feature(self, rgb: torch.Tensor, hsi: torch.Tensor, *args,
                     save_dir: str='SpectralFusion_features', **kwargs) -> None:

        mat_mode = kwargs.get('mat_mode', False)
        color_mode = kwargs.get('color_mode', False)
        save_color_dir = kwargs.get('save_color_dir', 'RGBHSCNN_color_features')
        data_name = kwargs.get('data_name', 'CAVE')
        pick_layers = kwargs.get('pick_layers', ['start_conv'] + list(self.activations.keys()) + ['output_conv'])
        row, col = int(np.ceil(np.sqrt(self.output_ch))), int(np.ceil(np.sqrt(self.output_ch)))
        os.makedirs(save_dir, exist_ok=True)
        if color_mode is True:
            os.makedirs(save_color_dir, exist_ok=True)
        features = self.get_feature(x, pick_layers)
        for layer_name, feature in features.items():
            # nd_feature = feature.squeeze().detach().numpy().transpose(1, 2, 0)
            feature = normalize(feature.permute((1, 0, 2, 3)))  # .clamp(0., 1.)
            torchvision.utils.save_image(feature, os.path.join(save_dir, f'{layer_name}.png'),
                                         nrow=row, padding=0)
            nd_feature = feature.squeeze().detach().numpy().transpose(1, 2, 0)
            if mat_mode is True:
                scipy.io.savemat(os.path.join(save_dir, f'{layer_name}.mat'), {'data': nd_feature})
            if color_mode is True:
                self.plot_color_img(nd_feature, os.path.join(save_color_dir, f'{layer_name}.png'),
                                    mode=data_name)

    def plot_diff(self, x: torch.Tensor, label: torch.Tensor, *args,
                  save_dir: str='HSCNN_diff', **kwargs) -> None:

        mat_mode = kwargs.get('mat_mode', False)
        pick_layers = kwargs.get('pick_layers', ['start_conv'] + list(self.activations.keys()) + ['output_conv'])
        _, ch, h, w = label.shape
        row, col = int(np.ceil(np.sqrt(self.output_ch))), int(np.ceil(np.sqrt(self.output_ch)))
        plot_array = np.zeros((h * row, col * w))
        os.makedirs(save_dir, exist_ok=True)
        features = self.get_feature(x, pick_layers)
        for layer_name, feature in features.items():
            plot_array[:] = 0.
            feature = (feature.clamp(0., 1.) - label).abs()
            feature_mean = feature.mean(dim=(-1, -2))
            nd_feature = feature.squeeze().detach().numpy().transpose(1, 2, 0)
            for i in range(row):
                for j in range(col):
                    if i * row + j >= ch: break
                    plot_array[h * i: h * (i + 1), w * j: w * (j + 1)] = nd_feature[:, :, i * row + j]
            plt.imsave(os.path.join(save_dir, f'{layer_name}.png'), plot_array, cmap='jet')
            feature_mean = feature_mean.squeeze().detach().numpy()
            if mat_mode is True:
                scipy.io.savemat(os.path.join(save_dir, f'{layer_name}.mat'), {'data': nd_feature, 'mean': feature_mean})
        plot_label = label.permute((1, 0, 2, 3))
        torchvision.utils.save_image(plot_label, os.path.join(save_dir, 'output_img.png'),
                                        nrow=row, padding=0)

    def plot_color_img(self, input_img: np.ndarray,
                       save_name: str, *args, mode='CAVE', **kwargs):
        func_name = transformations.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs
        if mode == 'CAVE' or mode == 'ICVL':
            start_wave = 400
            last_wave = 700
        else:
            start_wave = 420
            last_wave = 720
        x = np.arange(start_wave, last_wave + 1, 10)
        trans_filter = func_name(x)
        ch = x.shape[0]
        row = int(np.ceil(np.sqrt(ch)))
        all_img = []
        for i, ch in enumerate(range(start_wave, last_wave + 1, 10)):
            trans_data = np.expand_dims(input_img[:, :, i], axis=-1).dot(np.expand_dims(trans_filter[i], axis=0)).clip(0., 1.)
            all_img.append(trans_data)
        tensor_img = torch.Tensor(np.array(all_img)).permute(0, 3, 1, 2)
        torchvision.utils.save_image(tensor_img, save_name, nrow=row, padding=0)


class HSIHSCNN(Base_Module):

    def __init__(self, input_ch: int, output_ch: int, *args, feature_num: int=64,
                 layer_num: int=3, **kwargs) -> None:
        super().__init__()
        self.output_ch = output_ch
        activation = kwargs.get('activation', 'relu').lower()
        self.input_conv = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)
        self.input_activation = self.activations[activation]()
        self.feature_layers = torch.nn.ModuleDict({f'HSI_{i}': torch.nn.Conv2d(feature_num, feature_num, 3, 1, 1)
                                                   for i in range(layer_num)})
        self.activation_layer = torch.nn.ModuleDict({f'HSI_act_{i}': self.activations[activation]()
                                                     for i in range(layer_num)})
        self.output_conv = torch.nn.Conv2d(feature_num, output_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input_activation(self.input_conv(x))
        for layer, activation in zip(self.feature_layers.values(), self.activation_layer.values()):
            x = activation(layer(x))
        x = self.output_conv(x)
        return x

    def get_feature(self, x: torch.Tensor, pick_layer: list) -> dict:

        return_features = {}
        x = self.input_conv(x)
        x_in = x
        if 'start_conv' in pick_layer:
            return_features['start_conv'] = x_in
        x = self.input_activation(x)
        for (layer_name, layer), (activation_name, activation) in zip(self.feature_layers.items(), self.activation_layer.items()):
            x = activation(layer(x))
            if activation_name in pick_layer:
                return_features[activation_name] = x
        '''
        if self.residual:
            output = self.residual_conv(x) + x_in
        else:
            output = self.residual_conv(x)
        '''
        output = self.output_conv(x)
        if 'output_conv' in pick_layer:
            return_features['output_conv'] = output
        return return_features

    def plot_feature(self, rgb: torch.Tensor, hsi: torch.Tensor, *args,
                     save_dir: str='SpectralFusion_features', **kwargs) -> None:

        mat_mode = kwargs.get('mat_mode', False)
        color_mode = kwargs.get('color_mode', False)
        save_color_dir = kwargs.get('save_color_dir', 'HSIHSCNN_color_features')
        data_name = kwargs.get('data_name', 'CAVE')
        pick_layers = kwargs.get('pick_layers', ['start_conv'] + list(self.activations.keys()) + ['output_conv'])
        row, col = int(np.ceil(np.sqrt(self.output_ch))), int(np.ceil(np.sqrt(self.output_ch)))
        os.makedirs(save_dir, exist_ok=True)
        if color_mode is True:
            os.makedirs(save_color_dir, exist_ok=True)
        features = self.get_feature(x, pick_layers)
        for layer_name, feature in features.items():
            # nd_feature = feature.squeeze().detach().numpy().transpose(1, 2, 0)
            feature = normalize(feature.permute((1, 0, 2, 3)))  # .clamp(0., 1.)
            torchvision.utils.save_image(feature, os.path.join(save_dir, f'{layer_name}.png'),
                                         nrow=row, padding=0)
            nd_feature = feature.squeeze().detach().numpy().transpose(1, 2, 0)
            if mat_mode is True:
                scipy.io.savemat(os.path.join(save_dir, f'{layer_name}.mat'), {'data': nd_feature})
            if color_mode is True:
                self.plot_color_img(nd_feature, os.path.join(save_color_dir, f'{layer_name}.png'),
                                    mode=data_name)

    def plot_diff(self, x: torch.Tensor, label: torch.Tensor, *args,
                  save_dir: str='HSCNN_diff', **kwargs) -> None:

        mat_mode = kwargs.get('mat_mode', False)
        pick_layers = kwargs.get('pick_layers', ['start_conv'] + list(self.activations.keys()) + ['output_conv'])
        _, ch, h, w = label.shape
        row, col = int(np.ceil(np.sqrt(self.output_ch))), int(np.ceil(np.sqrt(self.output_ch)))
        plot_array = np.zeros((h * row, col * w))
        os.makedirs(save_dir, exist_ok=True)
        features = self.get_feature(x, pick_layers)
        for layer_name, feature in features.items():
            plot_array[:] = 0.
            feature = (feature.clamp(0., 1.) - label).abs()
            feature_mean = feature.mean(dim=(-1, -2))
            nd_feature = feature.squeeze().detach().numpy().transpose(1, 2, 0)
            for i in range(row):
                for j in range(col):
                    if i * row + j >= ch: break
                    plot_array[h * i: h * (i + 1), w * j: w * (j + 1)] = nd_feature[:, :, i * row + j]
            plt.imsave(os.path.join(save_dir, f'{layer_name}.png'), plot_array, cmap='jet')
            feature_mean = feature_mean.squeeze().detach().numpy()
            if mat_mode is True:
                scipy.io.savemat(os.path.join(save_dir, f'{layer_name}.mat'), {'data': nd_feature, 'mean': feature_mean})
        plot_label = label.permute((1, 0, 2, 3))
        torchvision.utils.save_image(plot_label, os.path.join(save_dir, 'output_img.png'),
                                        nrow=row, padding=0)

    def plot_color_img(self, input_img: np.ndarray,
                       save_name: str, *args, mode='CAVE', **kwargs):
        func_name = transformations.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs
        if mode == 'CAVE' or mode == 'ICVL':
            start_wave = 400
            last_wave = 700
        else:
            start_wave = 420
            last_wave = 720
        x = np.arange(start_wave, last_wave + 1, 10)
        trans_filter = func_name(x)
        ch = x.shape[0]
        row = int(np.ceil(np.sqrt(ch)))
        all_img = []
        for i, ch in enumerate(range(start_wave, last_wave + 1, 10)):
            trans_data = np.expand_dims(input_img[:, :, i], axis=-1).dot(np.expand_dims(trans_filter[i], axis=0)).clip(0., 1.)
            all_img.append(trans_data)
        tensor_img = torch.Tensor(np.array(all_img)).permute(0, 3, 1, 2)
        torchvision.utils.save_image(tensor_img, save_name, nrow=row, padding=0)


class SpectralFusion(Base_Module):

    def __init__(self, input_hsi_ch: int, input_rgb_ch: int, output_hsi_ch: int,
                 output_rgb_ch: int, *args, rgb_feature: int=64, hsi_feature: int=64,
                 fusion_feature: int=64, layer_num: int=3, **kwargs) -> None:
        super().__init__()
        self.input_rgb_ch = input_rgb_ch
        self.output_rgb_ch = output_rgb_ch
        self.output_hsi_ch = output_hsi_ch
        self.layer_num = layer_num
        self.rgb_layer = RGBHSCNN(input_rgb_ch, output_rgb_ch, feature_num=rgb_feature,
                                  layer_num=layer_num)
        self.hsi_layer = HSIHSCNN(input_hsi_ch, output_hsi_ch, feature_num=hsi_feature,
                                  layer_num=layer_num)
        self.fusion_layer = torch.nn.ModuleDict({f'Fusion_{i}': torch.nn.Conv2d(rgb_feature + hsi_feature, fusion_feature, 1, 1, 0)
                                                for i in range(layer_num)})

    def forward(self, rgb: torch.Tensor, hsi: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        hsi_x = self.hsi_layer.input_activation(self.hsi_layer.input_conv(hsi))
        if self.input_rgb_ch >= 1:
            rgb_x = self.rgb_layer.input_activation(self.rgb_layer.input_conv(rgb))
        else:
            rgb_x = hsi_x
        for i in range(self.layer_num):
            rgb_x = self.rgb_layer.activation_layer[f'RGB_act_{i}'](self.rgb_layer.feature_layers[f'RGB_{i}'](rgb_x))
            fusion_feature = torch.cat((rgb_x, hsi_x), dim=1)
            hsi_x = self.fusion_layer[f'Fusion_{i}'](fusion_feature)
            hsi_x = self.hsi_layer.activation_layer[f'HSI_act_{i}'](self.hsi_layer.feature_layers[f'HSI_{i}'](hsi_x))
        output_hsi = self.hsi_layer.output_conv(hsi_x)
        if self.output_rgb_ch >= 1:
            output_rgb = self.rgb_layer.output_conv(rgb_x)
            return output_rgb, output_hsi
        else:

            return output_hsi

    def get_feature(self, rgb: torch.Tensor, hsi: torch.Tensor, pick_layer: list) -> dict:

        return_features = {}
        hsi_x = self.hsi_layer.input_activation(self.hsi_layer.input_conv(hsi))
        if self.input_rgb_ch >= 1:
            rgb_x = self.rgb_layer.input_activation(self.rgb_layer.input_conv(rgb))
        else:
            rgb_x = hsi_x
        for i in range(self.layer_num):
            rgb_x = self.rgb_layer.activation_layer[f'RGB_act_{i}'](self.rgb_layer.feature_layers[f'RGB_{i}'](rgb_x))
            fusion_feature = torch.cat((rgb_x, hsi_x), dim=1)
            hsi_x = self.fusion_layer[f'Fusion_{i}'](fusion_feature)
            if f'Fusion_{i}' in pick_layer:
                return_features[f'Fusion_{i}'] = hsi_x
            hsi_x = self.hsi_layer.activation_layer[f'HSI_act_{i}'](self.hsi_layer.feature_layers[f'HSI_{i}'](hsi_x))
        return return_features

    def plot_feature(self, rgb: torch.Tensor, hsi: torch.Tensor, *args,
                     save_dir: str='SpectralFusion_features', **kwargs) -> None:

        mat_mode = kwargs.get('mat_mode', False)
        rgb_layers = kwargs.get('rgb_layers', ['start_conv'] + list(self.rgb_layer.activation_layer.keys()) + ['output_conv'])
        hsi_layers = kwargs.get('hsi_layers', ['start_conv'] + list(self.hsi_layer.activation_layer.keys()) + ['output_conv'])
        fusion_layers = kwargs.get('fusion_layers', list(self.fusion_layer.keys()))
        _, ch, h, w = label.shape
        row, col = int(np.ceil(np.sqrt(self.output_hsi_ch))), int(np.ceil(np.sqrt(self.output_hsi_ch)))
        plot_array = np.zeros((h * row, col * w))
        os.makedirs(save_dir, exist_ok=True)
        rgb_features = self.rgb_layer.get_feature(rgb, rgb_layers)
        hsi_features = self.hsi_layer.get_feature(hsi, hsi_layers)
        fusion_features = self.fusion_layer.get_feature(fusion, fusion_layers)
        features = {**rgb_features, **hsi_features, **fusion_features}
        for layer_name, feature in features.items():
            # nd_feature = feature.squeeze().detach().numpy().transpose(1, 2, 0)
            feature = normalize(feature.permute((1, 0, 2, 3)))  # .clamp(0., 1.)
            torchvision.utils.save_image(feature, os.path.join(save_dir, f'{layer_name}.png'),
                                         nrow=row, padding=0)
            nd_feature = feature.squeeze().detach().numpy().transpose(1, 2, 0)
            if mat_mode is True:
                scipy.io.savemat(os.path.join(save_dir, f'{layer_name}.mat'), {'data': nd_feature})
            if color_mode is True:
                self.plot_color_img(nd_feature, os.path.join(save_color_dir, f'{layer_name}.png'),
                                    mode=data_name)

    def plot_diff(self, rgb: torch.Tensor, hsi: torch.Tensor, label: torch.Tensor, *args,
                  save_dir: str='HSCNN_diff', **kwargs) -> None:

        mat_mode = kwargs.get('mat_mode', False)
        rgb_layers = kwargs.get('rgb_layers', ['start_conv'] + list(self.rgb_layer.activation_layer.keys()) + ['output_conv'])
        hsi_layers = kwargs.get('hsi_layers', ['start_conv'] + list(self.hsi_layer.activation_layer.keys()) + ['output_conv'])
        fusion_layers = kwargs.get('fusion_layers', list(self.fusion_layer.keys()))
        _, ch, h, w = label.shape
        row, col = int(np.ceil(np.sqrt(self.output_hsi_ch))), int(np.ceil(np.sqrt(self.output_hsi_ch)))
        plot_array = np.zeros((h * row, col * w))
        os.makedirs(save_dir, exist_ok=True)
        rgb_features = self.rgb_layer.get_feature(rgb, rgb_layers)
        hsi_features = self.hsi_layer.get_feature(hsi, hsi_layers)
        fusion_features = self.fusion_layer.get_feature(fusion, fusion_layers)
        features = {**rgb_features, **hsi_features, **fusion_features}
        for layer_name, feature in features.items():
            plot_array[:] = 0.
            feature = (feature.clamp(0., 1.) - label).abs()
            feature_mean = feature.mean(dim=(-1, -2))
            nd_feature = feature.squeeze().detach().numpy().transpose(1, 2, 0)
            for i in range(row):
                for j in range(col):
                    if i * row + j >= ch: break
                    plot_array[h * i: h * (i + 1), w * j: w * (j + 1)] = nd_feature[:, :, i * row + j]
            plt.imsave(os.path.join(save_dir, f'{layer_name}.png'), plot_array, cmap='jet')
            feature_mean = feature_mean.squeeze().detach().numpy()
            if mat_mode is True:
                scipy.io.savemat(os.path.join(save_dir, f'{layer_name}.mat'), {'data': nd_feature, 'mean': feature_mean})
        plot_label = label.permute((1, 0, 2, 3))
        torchvision.utils.save_image(plot_label, os.path.join(save_dir, 'output_img.png'),
                                        nrow=row, padding=0)

    def plot_color_img(self, input_img: np.ndarray,
                       save_name: str, *args, mode='CAVE', **kwargs):
        func_name = transformations.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs
        if mode == 'CAVE' or mode == 'ICVL':
            start_wave = 400
            last_wave = 700
        else:
            start_wave = 420
            last_wave = 720
        x = np.arange(start_wave, last_wave + 1, 10)
        trans_filter = func_name(x)
        ch = x.shape[0]
        row = int(np.ceil(np.sqrt(ch)))
        all_img = []
        for i, ch in enumerate(range(start_wave, last_wave + 1, 10)):
            trans_data = np.expand_dims(input_img[:, :, i], axis=-1).dot(np.expand_dims(trans_filter[i], axis=0)).clip(0., 1.)
            all_img.append(trans_data)
        tensor_img = torch.Tensor(np.array(all_img)).permute(0, 3, 1, 2)
        torchvision.utils.save_image(tensor_img, save_name, nrow=row, padding=0)

if __name__ == '__main__':

    model = HSCNN(1, 31, activation='sin2')
    summary(model, (1, 64, 64))

# coding: utf-8


import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchinfo import summary
from colour.colorimetry import transformations
from .layers import Base_Module, EDSR_Block, HSI_EDSR_Block, Ghost_Mix, Conv2d


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


class RGBHSCNN(Base_Module):

    '''
    RGB画像を再構成する補助タスク学習モデル

    Attributes:
        input_conv (torch.nn.Module):
            入力時の畳み込み処理. RGB画像を入力するときに使用する. 圧縮画像のみの場合は
            torch.nn.Identity()モジュールを使用する. 
        input_activation (torch.nn.Module):
            input_conv のあとの活性化関数. 使用する関数は__init__()メソッドの引数
            activation で決められる. 
        feature_layers (torch.nn.ModuleDict[str, torch.nn.Module]):
            補助タスクモデルの中間層. デフォルトでは3 x 3 の畳み込み層を使用する. 
            使用する処理層は__init__()メソッドの引数 rgb_mode で変更可能. 
            self.feature_layers['RGB_層の番号']で各層にアクセス可能
        output_conv (torch.nn.Module):
            出力処理層. RGB画像の3⃣チャンネルに調整する畳み込み処理を行う. 
    '''


    def __init__(self, input_ch: int, output_ch: int, *args, feature_num: int=64,
                 layer_num: int=3, rgb_mode='normal', **kwargs) -> None:

        '''
        Args:
            input_ch (int):
                入力画像のチャンネル数. 0を入力した場合スナップショット圧縮画像のみの
                入力となる. 
            output_ch (int):
                出力画像のチャンネル数. 
            feature_num (int, optional):
                中間層のチャンネル数. Default = 64
            layer_num (int, optional):
                中間層の数.  Default = 3
            rgb_moode (str, optional):
                中間層に使用する処理層. 'normal' の場合は torch.nn.Conv2d, 
                'edsr' の場合は Conv -> ReLU -> Conv, 'ghost' の場合は軽量畳み込み層(Ghost Net
                にて提案された畳み込み層)を使用できる. Default = 'normal
            activation (str, optional):
                入力層, 中間層に使用する活性化関数.  ReLU('relu'), Leaky ReLU('leaky'), 
                Swish('swish'), Mish('mish') から選択可能. 
                Default = 'relu'
            ratio (int, optional):
                軽量畳み込みそうを使用するときの圧縮率. 
                Default = 2
            edsr_mode (str, optional),:
                rgb_mode にて edsr を選択した場合に使用する畳み込み処理を選択する.
                'normal' の場合 Conv -> ReLU -> Conv, 'separable' の場合
                Depthwise Separable(DS) -> ReLU -> DS, 'ghost' の場合
                軽量畳み込み -> ReLU -> 軽量畳み込み となる.
                Default = 'normal'

        Returns:
            NoneType
        '''
        
        super().__init__()
        rgb_layer = {'normal': Conv2d, 'edsr': EDSR_Block, 'ghost': Ghost_Mix}
        activation = kwargs.get('activation', 'relu').lower()
        ratio = kwargs.get('ratio', 2)
        edsr_mode = kwargs.get('edsr_mode', 'normal')
        self.input_activation = self.activations[activation]()
        self.output_ch = output_ch
        # 圧縮画像を入力
        if input_ch == 0:
            self.input_conv = torch.nn.Identity()
        # RGB画像を入力
        else:
            self.input_conv = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)
        # 中間層
        self.feature_layers = torch.nn.ModuleDict({f'RGB_{i}': rgb_layer[rgb_mode.lower()](feature_num, feature_num, ratio=ratio, edsr_mode=edsr_mode)
                                                   for i in range(layer_num)})
        # 中間層に使用する活性化関数
        self.activation_layer = torch.nn.ModuleDict({f'RGB_act_{i}': self.activations[activation]()
                                                     for i in range(layer_num)})
        self.output_conv = torch.nn.Conv2d(feature_num, output_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        '''
        RGB画像の再構成処理

        Args:
            x (torch.Tensor): 入力画像. (B, C, H, W) の形状にする必要あり.

        Returns:
            output_img (torch.Tensor): 出力画像.
        '''

        x = self.input_activation(self.input_conv(x))
        # 各中間層処理を繰り返す
        for layer, activation in zip(self.feature_layers.values(), self.activation_layer.values()):
            x = activation(layer(x))
        output_img = self.output_conv(x)
        return output_img


class HSIHSCNN(Base_Module):

    '''
    ハイパースペクトル画像(HSI)を再構成する学習モデル

    Attributes:
        input_conv (torch.nn.Module):
            入力時の畳み込み処理. 
        input_activation (torch.nn.Module):
            input_conv のあとの活性化関数. 使用する関数は__init__()メソッドの引数
            activation で決められる. 
        feature_layers (torch.nn.ModuleDict[str, torch.nn.Module]):
            補助タスクモデルの中間層. デフォルトでは3 x 3 の畳み込み層を使用する. 
            使用する処理層は__init__()メソッドの引数 rgb_mode で変更可能. 
            self.feature_layers['HSI_層の番号']で各層にアクセス可能
        output_conv (torch.nn.Module):
            出力処理層. HSIのチャンネル数に調整する．

    '''


    def __init__(self, input_ch: int, output_ch: int, *args, feature_num: int=64,
                 layer_num: int=3, hsi_mode='normal', **kwargs) -> None:

        '''
        Args:
            input_ch (int):
                入力画像のチャンネル数. 
            output_ch (int):
                出力画像のチャンネル数. 
            feature_num (int, optional):
                中間層のチャンネル数. Default = 64
            layer_num (int, optional):
                中間層の数.  Default = 3
            hsi_moode (str, optional):
                中間層に使用する処理層. 'normal' の場合は torch.nn.Conv2d, 
                'edsr' の場合は Conv -> ReLU -> Conv, 'ghost' の場合は軽量畳み込み層(Ghost Net
                にて提案された畳み込み層)を使用できる. Default = 'normal
            activation (str, optional):
                入力層, 中間層に使用する活性化関数.  ReLU('relu'), Leaky ReLU('leaky'), 
                Swish('swish'), Mish('mish') から
                選択可能. Default = 'relu'
            ratio (int, optional):
                軽量畳み込みそうを使用するときの圧縮率. 
                Default = 2
            edsr_mode (str, optional),:
                hsi_mode にて edsr を選択した場合に使用する畳み込み処理を選択する.
                'normal' の場合 Conv -> ReLU -> Conv, 'separable' の場合
                Depthwise Separable(DS) -> ReLU -> DS, 'ghost' の場合
                軽量畳み込み -> ReLU -> 軽量畳み込み となる.
                Default = 'normal'

        Returns:
            NoneType
        '''
        
        super().__init__()
        hsi_layer = {'normal': Conv2d, 'edsr': EDSR_Block, 'ghost': Ghost_Mix}
        activation = kwargs.get('activation', 'relu').lower()
        ratio = kwargs.get('ratio', 2)
        edsr_mode = kwargs.get('edsr_mode', 'normal')
        self.input_activation = self.activations[activation]()
        self.output_ch = output_ch
        # 圧縮画像を入力
        if input_ch == 0:
            self.input_conv = torch.nn.Identity()
        # HSI画像を入力
        else:
            self.input_conv = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)
        # 中間層
        self.feature_layers = torch.nn.ModuleDict({f'HSI_{i}': hsi_layer[hsi_mode.lower()](feature_num, feature_num, ratio=ratio, edsr_mode=edsr_mode)
                                                   for i in range(layer_num)})
        # 中間層に使用する活性化関数
        self.activation_layer = torch.nn.ModuleDict({f'HSI_act_{i}': self.activations[activation]()
                                                     for i in range(layer_num)})
        self.output_conv = torch.nn.Conv2d(feature_num, output_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        '''
        HSIの再構成処理

        Args:
            x (torch.Tensor): 入力画像. (B, C, H, W) の形状にする必要あり.

        Returns:
            output_img (torch.Tensor): 出力画像.
        '''

        x = self.input_activation(self.input_conv(x))
        # 各中間層処理を繰り返す
        for layer, activation in zip(self.feature_layers.values(), self.activation_layer.values()):
            x = activation(layer(x))
        output_img = self.output_conv(x)
        return output_img


class SpectralFusion(Base_Module):

    '''
    RGB画像の再構成補助タスクを使用したハイパースペクトル(HSI)画像の再構成モデル


    Attributes:
        input_rgb_ch (int): RGB画像の入力チャンネル数
        output_rgb_ch (int): RGB画像の出力チャンネル数
        output_hsi_ch (int): HSI画像の出力チャンネル数
        layer_num (int): 中間層の数
        rgb_layer (torch.nn.Module): RGBの再構成補助タスクモデル. RGBHSCNNを使用
        hsi_layer (torch.nn.Module): HSIの再構成モデル. HSIHSCNNを使用
        fusion_layer (torch.nn.ModuleDict[str, torch.nn.ModuleDict]): 
            rgb_layer と hsi_layer から出力された特徴マップを融合させる処理層.
            (1 x 1) の畳み込み層を使用している．
    '''

    def __init__(self, input_hsi_ch: int, input_rgb_ch: int, output_hsi_ch: int,
                 output_rgb_ch: int, *args, rgb_feature: int=64, hsi_feature: int=64,
                 fusion_feature: int=64, layer_num: int=3, **kwargs) -> None:

        '''
        Parameters:
            input_rgb_ch (int): RGB画像の入力チャンネル数
            output_rgb_ch (int): RGB画像の出力チャンネル数
            input_hsi_ch (int): HSI画像の入力チャンネル数
            output_hsi_ch (int): HSI画像の出力チャンネル数
            rgb_feature (int, optional): RGBHSCNNにおける中間層のチャンネル数
                Default = 64
            hsi_feature (int, optional): HSIHSCNNにおける中間層のチャンネル数
                Default = 64
            fusion_feature (int, optional): fusion_layer のチャンネル数.  Default = 64
            layer_num (int, optional): 中間層の数
            activation (str, optional):
                入力層, 中間層に使用する活性化関数.  ReLU('relu'), Leaky ReLU('leaky'), 
                Swish('swish'), Mish('mish') から
                選択可能. Default = 'relu'
            rgb_moode (str, optional):
                RGBHSCNNにおける中間層に使用する処理層. 'normal' の場合は torch.nn.Conv2d, 
                'edsr' の場合は Conv -> ReLU -> Conv, 'ghost' の場合は軽量畳み込み層(Ghost Net
                にて提案された畳み込み層)を使用できる. Default = 'normal
            hsi_mode (str, optional):
                HSIHSCNNにおける中間層に使用する処理層. 'normal' の場合は torch.nn.Conv2d, 
                'edsr' の場合は Conv -> ReLU -> Conv, 'ghost' の場合は軽量畳み込み層(Ghost Net
                にて提案された畳み込み層)を使用できる. Default = 'normal
            edsr_mode (str, optional),:
                rgb_mode, hsi_mode にて edsr を選択した場合に使用する畳み込み処理を選択する.
                'normal' の場合 Conv -> ReLU -> Conv, 'separable' の場合
                Depthwise Separable(DS) -> ReLU -> DS, 'ghost' の場合
                軽量畳み込み -> ReLU -> 軽量畳み込み となる.
                Default = 'normal'
            rb_encoder_path (str, optional): 学習済みRGBHSCNNのファイル名. Default = None
        '''

        super().__init__()
        activation = kwargs.get('activation', 'relu')
        ratio = kwargs.get('ratio', 2)
        rgb_mode = kwargs.get('rgb_mode', 'normal')
        hsi_mode = kwargs.get('hsi_mode', 'normal')
        edsr_mode = kwargs.get('edsr_mode', 'normal')
        rgb_encoder_path = kwargs.get('rgb_encoder_path', None)
        self.input_rgb_ch = input_rgb_ch
        self.output_rgb_ch = output_rgb_ch
        self.output_hsi_ch = output_hsi_ch
        self.layer_num = layer_num
        self.mode = mode
        # RGB補助タスクモデル
        self.rgb_layer = RGBHSCNN(input_rgb_ch, output_rgb_ch, feature_num=rgb_feature,
                                  layer_num=layer_num, rgb_mode=rgb_mode, ratio=ratio, 
                                  edsr_mode=edsr_mode)
        # RGBHSCNNの学習済みモデルを読み込む
        if rgb_encoder_path is not None and isinstance(rgb_encoder_path, str):
            ckpt = torch.load(rgb_encoder_path, map_location='cpu')
            self.rgb_layer.load_state_dict(ckpt['model_state_dict'])
            # パラメータの固定(学習させない)
            for param in self.rgb_layer.parameters():
                param.requires_grad = False
        # HSI再構成モデル
        self.hsi_layer = HSIHSCNN(input_hsi_ch, output_hsi_ch, feature_num=hsi_feature,
                                  layer_num=layer_num, hsi_mode=hsi_mode, ratio=ratio, 
                                  edsr_mode=edsr_mode)
        # RGB特徴マップとHSI特徴マップの融合層
        self.fusion_layer = torch.nn.ModuleDict({f'Fusion_{i}': torch.nn.Conv2d(rgb_feature + hsi_feature, fusion_feature, 1, 1, 0)
                                                 for i in range(layer_num)})

    def forward(self, rgb: torch.Tensor, hsi: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        '''
        実際に再構成を行う

        Args:
            rgb (torch.Tensor): 入力となるRGB画像
            hsi (torch.Tensor): 入力となるスナップショット画像

        Returns:
            output_hsi (torch.Tensor): 出力となるHSI

        '''

        # スナップショット画像のスペクトル成分を拡張
        hsi_x = self.hsi_layer.input_activation(self.hsi_layer.input_conv(hsi))
        if self.input_rgb_ch > 1:
            # RGB画像のチャンネル数を拡張
            rgb_x = self.rgb_layer.input_activation(self.rgb_layer.input_conv(rgb))
        else:
            # スナップショット画像のスペクトル成分を拡張
            rgb_x = self.rgb_layer.input_activation(self.rgb_layer.input_conv(hsi))
        # RGB再構成処理とHSI再構成処理を繰り返す
        for i in range(self.layer_num):
            # RGB再構成処理
            rgb_x = self.rgb_layer.activation_layer[f'RGB_act_{i}'](self.rgb_layer.feature_layers[f'RGB_{i}'](rgb_x))
            # RGB特徴マップとHSI特徴マップを融合
            fusion_feature = torch.cat((rgb_x, hsi_x), dim=1)
            fusion_feature = self.fusion_layer[f'Fusion_{i}'](fusion_feature)
            # 融合後の特徴マップをHSIとして再構成
            hsi_x = fusion_feature
            hsi_x = self.hsi_layer.activation_layer[f'HSI_act_{i}'](self.hsi_layer.feature_layers[f'HSI_{i}'](hsi_x))
        # チャンネル数の調整
        output_hsi = self.hsi_layer.output_conv(hsi_x)
        # 出力画像にRGB画像を含む場合
        if self.output_rgb_ch >= 1:
            output_rgb = self.rgb_layer.output_conv(rgb_x)
            return output_rgb, output_hsi
        # 出力画像にRGB画像を含まない場合
        else:
            return output_hsi

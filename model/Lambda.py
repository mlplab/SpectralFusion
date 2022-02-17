# coding: UTF-8


import torch


class UNetConv2d(torch.nn.Module):

    '''
    UNet に使用する畳み込み層
    ReLU -> Conv の処理をまとめている
    
    Attributes:
        activation (torch.nn.Module): 活性化関数. 今回はReLU関数を使用.
        conv (torch.nn.Module): 畳み込み層
    '''

    def __init__(self, input_ch: int, output_ch: int, *args, 
            kernel_size: int=3, stride: int=1, padding: bool=None, **kwargs) -> None:

        '''
        Args:
            input_ch (int): 入力チャンネル
            output_ch (int): 出力チャンネル
            kernel_size (int, optional): 畳み込みに使用するカーネル数. Default = 3
            stride (int, optional): 畳み込みのストライド. Default = 1
            padding (int, optional): パディングのオプション. Default = False
        Returns:
            None
        '''
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.activation = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(input_ch, output_ch, kernel_size, stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        '''
        実際の処理
        Args:
            x (torch.Tensor): 入力の特徴マップ
        Returns:
            output (torch.Tensor): 出力の特徴マップ
        '''
        output = self.activation(self.conv(x))
        return output


class UNet(torch.nn.Module):

    '''
    UNet の処理

    Attributes:
        skip_encode_feature (list[int]): encoder 層のスキップ接続する層番号
        input_conv (torch.nn.Module): 入力時の畳み込み層
        encoder (torch.nn.ModuleList[torch.nn.Module]): encoder層
        bottleneck1 (torch.nn.Module): ボトルネック層1
        bottleneck2 (torch.nn.Module): ボトルネック層2
        skip_decode_feature (list[int]): decoder 層のスキップ接続する層番号
        decoder (torch.nn.ModuleList[torch.nn.Module]): decoder層
        output_conv (torch.nn.Module): 出力時の畳み込み層
    '''
    def __init__(self, input_ch: int, output_ch: int, *args, layer_num: int=6, deeper: bool=False, **kwargs) -> None:

        '''
        Args:
            input_ch (int):
                入力画像のチャンネル数. 
            output_ch (int):
                出力画像のチャンネル数. 
            layer_num (int, optional):
                中間層の数.  Default = 3
            deeper (bool, optional):
                Lambda-Net の Deeper オプション(未実装)
                Default = False
            activation (str, opitonal): 活性化関数. Default = 'relu'
        Returns:
            None
        '''
        super().__init__()
        activation = kwargs.get('activation', 'relu').lower()
        encode_feature_num = [2 ** (5 + i) for i in range(layer_num)]
        decode_feature_num = encode_feature_num[::-1]
        self.deeper = deeper
        self.skip_encode_feature = [2, 7, 12, 17, 21]
        self.input_conv = UNetConv2d(input_ch, encode_feature_num[0], 3, 1)
        self.encoder = torch.nn.ModuleList([
                UNetConv2d(encode_feature_num[0], encode_feature_num[0], 3, 1),
                UNetConv2d(encode_feature_num[0], encode_feature_num[0], 3, 1),
                UNetConv2d(encode_feature_num[0], encode_feature_num[0], 3, 1),
                torch.nn.MaxPool2d((2, 2)),
                UNetConv2d(encode_feature_num[0], encode_feature_num[1], 3, 1),
                UNetConv2d(encode_feature_num[1], encode_feature_num[1], 3, 1),
                UNetConv2d(encode_feature_num[1], encode_feature_num[1], 3, 1),
                UNetConv2d(encode_feature_num[1], encode_feature_num[1], 3, 1),
                torch.nn.MaxPool2d((2, 2)),
                UNetConv2d(encode_feature_num[1], encode_feature_num[2], 3, 1),
                UNetConv2d(encode_feature_num[2], encode_feature_num[2], 3, 1),
                UNetConv2d(encode_feature_num[2], encode_feature_num[2], 3, 1),
                UNetConv2d(encode_feature_num[2], encode_feature_num[2], 3, 1),
                torch.nn.MaxPool2d((2, 2)),
                UNetConv2d(encode_feature_num[2], encode_feature_num[3], 3, 1),
                UNetConv2d(encode_feature_num[3], encode_feature_num[3], 3, 1),
                UNetConv2d(encode_feature_num[3], encode_feature_num[3], 3, 1),
                UNetConv2d(encode_feature_num[3], encode_feature_num[3], 3, 1),
                torch.nn.MaxPool2d((2, 2)),
                UNetConv2d(encode_feature_num[3], encode_feature_num[4], 3, 1),
                UNetConv2d(encode_feature_num[4], encode_feature_num[4], 3, 1),
                UNetConv2d(encode_feature_num[4], encode_feature_num[4], 3, 1),
                torch.nn.MaxPool2d((2, 2)),
                UNetConv2d(encode_feature_num[4], encode_feature_num[5], 3, 1),
                ])
        self.bottleneck1 = UNetConv2d(encode_feature_num[-1], encode_feature_num[-1], 3, 1)
        self.bottleneck2 = UNetConv2d(encode_feature_num[-1], decode_feature_num[0], 3, 1)
        self.skip_decode_feature = [1, 4, 8, 12, 16]
        self.decoder = torch.nn.ModuleList([
                UNetConv2d(decode_feature_num[0], decode_feature_num[0], 3, 1),
                torch.nn.ConvTranspose2d(decode_feature_num[0], decode_feature_num[1], 2, 2),
                UNetConv2d(decode_feature_num[1] + encode_feature_num[-2],
                          decode_feature_num[1], 3, 1, 1),
                UNetConv2d(decode_feature_num[1], decode_feature_num[1], 3, 1),
                torch.nn.ConvTranspose2d(decode_feature_num[1], decode_feature_num[2], 2, 2),
                UNetConv2d(decode_feature_num[2] + encode_feature_num[-3],
                           decode_feature_num[2], 3, 1),
                UNetConv2d(decode_feature_num[2], decode_feature_num[2], 3, 1),
                UNetConv2d(decode_feature_num[2], decode_feature_num[2], 3, 1),
                torch.nn.ConvTranspose2d(decode_feature_num[2], decode_feature_num[3], 2, 2),
                UNetConv2d(decode_feature_num[3] + encode_feature_num[-4],
                           decode_feature_num[3], 3, 1),
                UNetConv2d(decode_feature_num[3], decode_feature_num[3], 3, 1),
                UNetConv2d(decode_feature_num[3], decode_feature_num[3], 3, 1),
                torch.nn.ConvTranspose2d(decode_feature_num[3], decode_feature_num[4], 2, 2),
                UNetConv2d(decode_feature_num[4] + encode_feature_num[-5],
                           decode_feature_num[4], 3, 1),
                UNetConv2d(decode_feature_num[4], decode_feature_num[4], 3, 1),
                UNetConv2d(decode_feature_num[4], decode_feature_num[4], 3, 1),
                torch.nn.ConvTranspose2d(decode_feature_num[4], decode_feature_num[5], 2, 2),
                UNetConv2d(decode_feature_num[5] + encode_feature_num[-6],
                           decode_feature_num[5], 3, 1),
                UNetConv2d(decode_feature_num[5], decode_feature_num[5], 3, 1),
                UNetConv2d(decode_feature_num[5], decode_feature_num[5], 3, 1),
                ])
        self.output_conv = UNetConv2d(decode_feature_num[-1], output_ch, 3, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        '''
        実際の処理
        Args:
            x (torch.Tensor): 入力画像
        Returns:
            output (torch.Tensor): 出力画像
        '''
        x = self.input_conv(x)
        encode_feature = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.skip_encode_feature:
                encode_feature.append(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        j = 1
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i in self.skip_decode_feature:
                x = torch.cat([x, encode_feature[-j]], dim=1)
                j += 1
        output = self.output_conv(x)

        return output


class RefineUNet(torch.nn.Module):

    '''
    Lambda-Net におけるRefineNet の処理

    Attributes:
        skip_encode_feature (list[int]): encoder 層のスキップ接続する層番号
        input_conv (torch.nn.Module): 入力時の畳み込み層
        encoder (torch.nn.ModuleList[torch.nn.Module]): encoder層
        bottleneck1 (torch.nn.Module): ボトルネック層1
        bottleneck2 (torch.nn.Module): ボトルネック層2
        skip_decode_feature (list[int]): decoder 層のスキップ接続する層番号
        decoder (torch.nn.ModuleList[torch.nn.Module]): decoder層
        output_conv (torch.nn.Module): 出力時の畳み込み層
    '''

    def __init__(self, input_ch: int, output_ch: int, *args, layer_num: int=4, deeper: bool=False, **kwargs):

        '''
        Args:
            input_ch (int):
                入力画像のチャンネル数. 
            output_ch (int):
                出力画像のチャンネル数. 
            layer_num (int, optional):
                中間層の数.  Default = 3
            deeper (bool, optional):
                Lambda-Net の Deeper オプション(未実装)
                Default = False
            activation (str, opitonal): 活性化関数. Default = 'relu'
        Returns:
            None
        '''
        super().__init__()
        activation = kwargs.get('activation', 'relu').lower()
        encode_feature_num = [2 ** (5 + i) for i in range(layer_num)]
        decode_feature_num = encode_feature_num[::-1]
        self.deeper = deeper
        self.skip_encode_feature = [2, 7, 12, 17, 21]
        self.input_conv = UNetConv2d(input_ch, encode_feature_num[0], 3, 1)
        self.encoder = torch.nn.ModuleList([
                UNetConv2d(encode_feature_num[0], encode_feature_num[0], 3, 1),
                UNetConv2d(encode_feature_num[0], encode_feature_num[0], 3, 1),
                UNetConv2d(encode_feature_num[0], encode_feature_num[0], 3, 1),
                torch.nn.MaxPool2d((2, 2)),
                UNetConv2d(encode_feature_num[0], encode_feature_num[1], 3, 1),
                UNetConv2d(encode_feature_num[1], encode_feature_num[1], 3, 1),
                UNetConv2d(encode_feature_num[1], encode_feature_num[1], 3, 1),
                UNetConv2d(encode_feature_num[1], encode_feature_num[1], 3, 1),
                torch.nn.MaxPool2d((2, 2)),
                UNetConv2d(encode_feature_num[1], encode_feature_num[2], 3, 1),
                UNetConv2d(encode_feature_num[2], encode_feature_num[2], 3, 1),
                UNetConv2d(encode_feature_num[2], encode_feature_num[2], 3, 1),
                UNetConv2d(encode_feature_num[2], encode_feature_num[2], 3, 1),
                torch.nn.MaxPool2d((2, 2)),
                UNetConv2d(encode_feature_num[2], encode_feature_num[3], 3, 1),
                ])
        self.bottleneck1 = UNetConv2d(encode_feature_num[-1], encode_feature_num[-1], 3, 1)
        self.bottleneck2 = UNetConv2d(encode_feature_num[-1], decode_feature_num[0], 3, 1)
        self.skip_decode_feature = [1, 4, 8, 12, 16]
        self.decoder = torch.nn.ModuleList([
                UNetConv2d(decode_feature_num[0], decode_feature_num[0], 3, 1),
                torch.nn.ConvTranspose2d(decode_feature_num[0], decode_feature_num[1], 2, 2),
                UNetConv2d(decode_feature_num[1] + encode_feature_num[-2],
                          decode_feature_num[1], 3, 1, 1),
                UNetConv2d(decode_feature_num[1], decode_feature_num[1], 3, 1),
                torch.nn.ConvTranspose2d(decode_feature_num[1], decode_feature_num[2], 2, 2),
                UNetConv2d(decode_feature_num[2] + encode_feature_num[-3],
                           decode_feature_num[2], 3, 1),
                UNetConv2d(decode_feature_num[2], decode_feature_num[2], 3, 1),
                UNetConv2d(decode_feature_num[2], decode_feature_num[2], 3, 1),
                torch.nn.ConvTranspose2d(decode_feature_num[2], decode_feature_num[3], 2, 2),
                UNetConv2d(decode_feature_num[3] + encode_feature_num[-4],
                           decode_feature_num[3], 3, 1),
                UNetConv2d(decode_feature_num[3], decode_feature_num[3], 3, 1),
                UNetConv2d(decode_feature_num[3], decode_feature_num[3], 3, 1)
                ])
        self.output_conv = UNetConv2d(decode_feature_num[-1], output_ch, 3, 1)


    def forward(self, x):
        '''
        実際の処理
        Args:
            x (torch.Tensor): 入力画像
        Returns:
            output (torch.Tensor): 出力画像
        '''
        x = self.input_conv(x)
        encode_feature = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.skip_encode_feature:
                encode_feature.append(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        j = 1
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i in self.skip_decode_feature:
                x = torch.cat([x, encode_feature[-j]], dim=1)
                j += 1
        output = self.output_conv(x)

        return output


class Discriminator(torch.nn.Module):

    '''
    GAN における Discriminator

    Attributes:
        conv_layers (torch.nn.ModuleList[torch.nn.Module]): 中間層
        fc (torch.nn.Module): 出力層
        output_activation (torch.nn.Module): 出力時の活性化関数. sigmoid 関数を使用.
    '''

    def __init__(self, input_ch: int, input_h: int, input_w: int, *args, layer_num: int=4, **kwargs) -> None:

        '''
        Args:
            input_ch (int): 入力チャンネル
            input_h (int): 入力画像の高さ(x軸)
            input_w (int): 入力画像の横幅(y軸)
            layer_num (int, optional): 中間層の数. Default = 4
        Returns
            None
        '''
        super().__init__()
        feature_num = [input_ch] + [2 ** (6 + i) for i in range(layer_num)]
        output_h, output_w = input_h // (2 ** (layer_num)), input_w // (2 ** (layer_num))
        self.conv_layers = torch.nn.ModuleList([UNetConv2d(feature_num[i], feature_num[i + 1], kernel_size=3, stride=2)
                                                for i in range(layer_num)])
        self.fc = torch.nn.Linear(output_h * output_w * feature_num[-1], 1)
        self.output_activation = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        識別処理

        Args:
            x (torch.Tensor): 入力画像
        Returns: 
            output(torch.Tensor): 出力値
        '''
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
        x = torch.flatten(x, 1)
        output = self.output_activation(self.fc(x))
        return output


class LambdaNet(torch.nn.Module):

    '''
    Lambda-Net
        Attributes:
            device (str): モデルを読み込む際に使用するデバイス (CPUに設定)
            ReconstStage (torch.nn.Module): ReconstStage に使用するモデル
            RefineStage (torch.nn.Module): RefineStage に使用するモデル
    '''

    def __init__(self, input_ch: int, output_ch: int, *args, layer_num: int=6, deeper: bool=False, **kwargs) -> None:

        '''
        Args:
            input_ch (int):
                入力画像のチャンネル数. 
            output_ch (int):
                出力画像のチャンネル数. 
            layer_num (int, optional):
                中間層の数.  Default = 3
            deeper (bool, optional):
                Lambda-Net の Deeper オプション(未実装)
                Default = False
        Returns:
            None
        '''

        super().__init__()
        self.device = kwargs.get('device', 'cpu')
        self.ReconstStage = UNet(input_ch, output_ch)
        self.RefineStage = RefineUNet(output_ch, output_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        実際の処理

        Args:
            x (torch.Tensor): 入力画像
        Returns (torch.Tensor): 出力画像
        '''
        return self.RefineStage(self.ReconstStage(x))

    def load_Reconst(self, path, key='model_state_dict') -> None:
        '''
        学習済みの self.ReconstStage をロード

        Args:
            path (str): 学習済みモデルの保存先
            key (str, optional): モデル読み込み時のkey. Default = 'model_state_dict'
        Returns:
            None
        '''
        ckpt = torch.load(path, map_location=self.device)
        self.ReconstStage.load_state_dict(ckpt[key])
        return self

    def load_Refine(self, path, key='model_state_dict') -> None:
        '''
        学習済みの self.RefineStage をロード

        Args:
            path (str): 学習済みモデルの保存先
            key (str, optional): モデル読み込み時のkey. Default = 'model_state_dict'
        Returns:
            None
        '''
        ckpt = torch.load(path, map_location=self.device)
        self.RefineStage.load_state_dict(ckpt[key])
        return self

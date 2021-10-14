# coding: utf-8


import os
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter
from model.RGBEncoder import SpectralFusionRGBEncoder
from model.SpectralFusion import SpectralFusion, RGBHSCNN


if os.path.exists('./runs'):
	shutil.rmtree('./runs')
writer = SummaryWriter('./runs/test_graph2')
# model = SpectralFusionRGBEncoder(1, 1, 31, 3)
model = SpectralFusion(input_rgb_ch=1, input_hsi_ch=1, output_hsi_ch=31, output_rgb_ch=3)
input_rgb = torch.rand((1, 1, 64, 64))
input_hsi = torch.rand((1, 1, 64, 64))
writer.add_graph(model, (input_rgb, input_hsi))
# model = RGBHSCNN(1, 3)
# writer.add_graph(model, input_rgb)
writer.close()

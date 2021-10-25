# coding: utf-8


import os
import sys
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchinfo import summary
from trainer import Trainer
from model.HSCNN import HSCNN
from model.DeepSSPrior import DeepSSPrior
from model.HyperReconNet import HyperReconNet
from model.SpectralFusion import SpectralFusion
from model.RGBEncoder import SpectralFusionRGBEncoder
from data_loader import PatchMaskDataset, PatchEvalDataset, SpectralFusionEvalDataset, RGBTrainEvalDataloader
from evaluate import PSNRMetrics, SAMMetrics, RMSEMetrics
from evaluate import ReconstEvaluater
from pytorch_ssim import SSIM
from utils import RandomCrop, RandomHorizontalFlip, RandomRotation
from utils import ModelCheckPoint, Draw_Output


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--batch_size', '-b', default=64, type=int, help='Training and validatio batch size')
parser.add_argument('--epochs', '-e', default=100, type=int, help='Train eopch size')
parser.add_argument('--dataset', '-d', default='Harvard', type=str, help='Select dataset')
parser.add_argument('--concat', '-c', default='False', type=str, help='Concat mask by input')
parser.add_argument('--model_name', '-m', default='SpectralFusionRGBEncoder', type=str, help='Model Name')
parser.add_argument('--block_num', '-bn', default=3, type=int, help='Model Block Number')
parser.add_argument('--start_time', '-st', default='0000', type=str, help='start training time')
parser.add_argument('--mode', '-md', default='inputOnly', type=str, help='Model Mode')
parser.add_argument('--loss', '-l', default='mse', type=str, help='Loss Mode')
parser.add_argument('--conv_mode', '-cm', default='normal', type=str, help='Conv Layer Mode')
parser.add_argument('--rgb_name', '-rn', default='RGBEncoder', type=str, help='PreTrain Model Name')
args = parser.parse_args()


dt_now = args.start_time
batch_size = args.batch_size
epochs = args.epochs
if args.concat == 'False':
    concat_flag = False
    input_ch = 1
else:
    concat_flag = True
    input_ch = 32
data_name = args.dataset
model_name = args.model_name
block_num = args.block_num
output_mode = args.mode
loss_mode = args.loss
conv_mode = args.conv_mode
rgb_name = args.rgb_name


device = 'cpu'


mode = {'both': [True, True, 'fusion', 3, 3],
        'inputOnly': [False, True, 'fusion', 0, 3],
        'outputOnly': [True, False, 'mse', 3, 0]}
load_mode = {'CAVE': 'mat',
             'Harvard': 'mat',
             'ICVL': 'mat'}
img_path = f'../SCI_dataset/My_{data_name}'
test_path = os.path.join(img_path, 'eval_data')
mask_path  = os.path.join(img_path, 'eval_mask_data')
sota_path = os.path.join('../SCI_ckpt', f'{data_name}_SOTA')
ckpt_path = os.path.join('../SCI_ckpt', f'{data_name}_{dt_now}')
all_trained_ckpt_path = os.path.join(ckpt_path, 'all_trained')
os.makedirs(all_trained_ckpt_path, exist_ok=True)


input_rgb = 32 if concat_flag else 1
output_rgb = 3
input_hsi = 32 if concat_flag else 1
output_hsi = 31


save_model_name = f'{model_name}_{block_num:02d}_{loss_mode}_{output_mode}_{dt_now}_{concat_flag}_{conv_mode}'


model_names = os.listdir(sota_path)
model_names = [name.split('.')[0] for name in model_names]


output_path = os.path.join('../SCI_result/', f'{data_name}_{dt_now}', save_model_name)
output_img_path = os.path.join(output_path, 'output_img')
output_mat_path = os.path.join(output_path, 'output_mat')
output_csv_path = os.path.join(output_path, f'output.csv')
output_fig_path = os.path.join(output_path, 'figure')
os.makedirs(output_path, exist_ok=True)
os.makedirs(output_img_path, exist_ok=True)
os.makedirs(output_mat_path, exist_ok=True)
os.makedirs(output_fig_path, exist_ok=True)
if os.path.exists(output_csv_path):
    print('already evaluated')
    sys.exit(0)


test_dataset = RGBTrainEvalDataloader(test_path, mask_path,
                                      transform=None, concat=concat_flag,
                                      data_name=data_name, rgb_input=mode[output_mode][0],
                                      rgb_label=mode[output_mode][1],
                                      load_mode=load_mode[data_name])


rgb_encoder_path = f'{rgb_name}_{block_num:02d}_{loss_mode}_{output_mode}_{dt_now}_{concat_flag}_{conv_mode}'

model = SpectralFusionRGBEncoder(input_rgb_ch=input_rgb, input_hsi_ch=input_hsi,
                                 output_rgb_ch=output_rgb, output_hsi_ch=output_hsi,
                                 rgb_feature=31, hsi_feature=31, fusion_feature=31,
                                 layer_num=block_num, rgb_mode=conv_mode, hsi_mode=conv_mode,
<<<<<<< HEAD
                                 rgb_encoder_path=os.path.join(all_trained_ckpt_path, f'{rgb_encoder_path}.tar'))
=======
                                 rgb_encoder_path=os.path.join(all_trained_ckpt_path, f'{rgb_encoder_path}.tar')).to(device)
>>>>>>> cab063e7adb6747480f19cc0f2da5f9b1560b90c


ckpt = torch.load(os.path.join(all_trained_ckpt_path, f'{save_model_name}.tar'),
                  map_location=torch.device('cpu'))
model.load_state_dict(ckpt['model_state_dict'])


<<<<<<< HEAD
model.to('cuda')
=======
model.to(device)
>>>>>>> cab063e7adb6747480f19cc0f2da5f9b1560b90c
# summary(model, (1, input_ch, 48, 48), depth=8)
psnr = PSNRMetrics().to(device).eval()
ssim = SSIM().to(device).eval()
sam = SAMMetrics().to(device).eval()
evaluate_fn = [psnr, ssim, sam]

evaluate = ReconstEvaluater(data_name, output_img_path, output_mat_path, output_csv_path)
evaluate.metrics(model, test_dataset, evaluate_fn, ['PSNR', 'SSIM', 'SAM'])

# coding: utf-8


import os
import sys
import argparse
import datetime
import torch
import torchvision
from torchinfo import summary
from trainer import Trainer
from model.HSCNN import HSCNN
from model.DeepSSPrior import DeepSSPrior
from model.HyperReconNet import HyperReconNet
from model.SpectralFusion import SpectralFusion, HSIHSCNN
from model.layers import MSE_SAMLoss, FusionLoss, RMSELoss, SpectralMSELoss
from data_loader import PatchMaskDataset, SpectralFusionDataset
from evaluate import PSNRMetrics, SAMMetrics
from utils import RandomCrop, RandomHorizontalFlip, RandomRotation
from utils import ModelCheckPoint, Draw_Output
from pytorch_ssim import SSIM


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--batch_size', '-b', default=64, type=int, help='Training and validatio batch size')
parser.add_argument('--epochs', '-e', default=100, type=int, help='Train eopch size')
parser.add_argument('--dataset', '-d', default='Harvard', type=str, help='Select dataset')
parser.add_argument('--concat', '-c', default='False', type=str, help='Concat mask by input')
parser.add_argument('--model_name', '-m', default='HSIHSCNN', type=str, help='Model Name')
parser.add_argument('--block_num', '-bn', default=3, type=int, help='Model Block Number')
parser.add_argument('--start_time', '-st', default='0000', type=str, help='start training time')
parser.add_argument('--mode', '-md', default='both', type=str, help='Model Mode')
parser.add_argument('--loss', '-l', default='mse', type=str, help='Loss Mode')
parser.add_argument('--conv_mode', '-cm', default='normal', type=str, help='Conv Layer Mode')
parser.add_argument('--edsr_mode', '-em', default='normal', type=str, help='Conv Layer Mode')
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True


mode = {'both': [True, True, 'fusion', 3, 3],
        'inputOnly': [False, True, 'fusion', 0, 3],
        'outputOnly': [True, False, 'mse', 3, 0]}
load_mode = {'CAVE': 'mat',
             'Harvard': 'mat',
             'ICVL': 'h5'}
img_path = f'../SCI_dataset/My_{data_name}'
train_path = os.path.join(img_path, 'train_patch_data')
test_path = os.path.join(img_path, 'test_patch_data')
mask_path = os.path.join(img_path, 'mask_data')
eval_mask_path = os.path.join(img_path, 'eval_mask_data')
callback_path = os.path.join(img_path, 'callback_path')
callback_mask_path = os.path.join(img_path, 'mask_show_data')
callback_result_path = os.path.join('../SCI_result', f'{data_name}_{dt_now}', f'{model_name}_{block_num}')
os.makedirs(callback_result_path, exist_ok=True)
filter_path = os.path.join('../SCI_dataset', 'D700_CSF.mat')
ckpt_path = os.path.join('../SCI_ckpt', f'{data_name}_{dt_now}')
all_trained_ckpt_path = os.path.join(ckpt_path, 'all_trained')
os.makedirs(all_trained_ckpt_path, exist_ok=True)


edsr_mode = args.edsr_mode
save_model_name = f'{model_name}_{block_num:02d}_{loss_mode}_{dt_now}_{concat_flag}_{conv_mode}_{edsr_mode}'
if os.path.exists(os.path.join(all_trained_ckpt_path, f'{save_model_name}.tar')):
    print(f'already trained {save_model_name}')
    sys.exit(0)


train_transform = (RandomHorizontalFlip(), torchvision.transforms.ToTensor())
test_transform = None
train_dataset = PatchMaskDataset(train_path, mask_path,
                                 transform=train_transform, concat=concat_flag,
                                 data_name=data_name)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=4)
test_dataset = PatchMaskDataset(test_path, eval_mask_path,
                                transform=test_transform, concat=concat_flag,
                                data_name=data_name)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=8)


model = HSIHSCNN(input_ch=input_ch, output_ch=31,
                 feature_num=31, layer_num=block_num, hsi_mode=conv_mode, edsr_mode=edsr_mode).to(device)
criterions = {'mse': torch.nn.MSELoss, 'rmse': RMSELoss, 'mse_sam': MSE_SAMLoss, 'fusion': FusionLoss, 'spectral': SpectralMSELoss}
criterion = criterions[loss_mode]().to(device)
param = list(model.parameters())
optim = torch.optim.Adam(lr=1e-3, params=param)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 25, .5)


trainer = Trainer(model, criterion, optim, scheduler=scheduler,
                  callbacks=None, device=device, use_amp=True,
                  psnr=PSNRMetrics(), ssim=SSIM(), sam=SAMMetrics())
train_loss, val_loss = trainer.train(epochs, train_dataloader, test_dataloader)
torch.save({'model_state_dict': model.state_dict(),
            'optim': optim.state_dict(),
            'train_loss': train_loss, 'val_loss': val_loss,
            'epoch': epochs},
           os.path.join(all_trained_ckpt_path, f'{save_model_name}.tar'))

# coding: utf-8


import os
import random
import shutil
import argparse
import scipy.io
import numpy as np
from tqdm import tqdm
from utils import make_patch, patch_mask, make_patch_list


patch_size = 64
patch_step = 64
show_size = 512
show_step = 512


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--dataset', '-d', default='Harvard', type=str, help='Select dataset')
args = parser.parse_args()


data_name = args.dataset
data_path = f'../SCI_dataset/{data_name}'
save_path = f'../SCI_dataset/My_{data_name}'


train_data_path = os.path.join(save_path, 'train_data')
test_data_path = os.path.join(save_path, 'test_data')


train_patch_path = os.path.join(save_path, 'train_patch_data')
test_patch_path = os.path.join(save_path, 'test_patch_data')
eval_path = os.path.join(save_path, 'eval_data')
eval_show_path = os.path.join(save_path, 'eval_show_data')


callback_path = os.path.join(save_path, 'callback_path')


mask_path = os.path.join(save_path, 'mask_data')
eval_mask_path = os.path.join(save_path, 'eval_mask_data')
mask_show_path = os.path.join(save_path, 'mask_show_data')


data_key = {'CAVE': 'im', 'Harvard': 'ref', 'ICVL': 'rad'}
data_size = {'CAVE': (512, 512, 31), 'Harvard': (1040, 1392, 31), 'ICVL': (1392, 1300, 31)}
seed_key = {'CAVE': 1, 'Harvard': 2, 'ICVL': 3}
np.random.seed(seed_key[data_name])


def move_data(data_path, data_list, move_path):
    os.makedirs(move_path, exist_ok=True)
    for name in tqdm(data_list, ascii=True):
        shutil.copy(os.path.join(data_path, name), os.path.join(move_path, name))
    return None


os.makedirs(save_path, exist_ok=True)
mask_data = np.random.choice((0., 1.),
                             (data_size[data_name][0] + data_size[data_name][2], data_size[data_name][1]),
                             p=(.5, .5))
mask_data = np.array([mask_data[i: data_size[data_name][0] + i] for i in range(data_size[data_name][-1])]).transpose(1, 2, 0)
scipy.io.savemat(os.path.join(save_path, 'mask.mat'), {'data': mask_data})
data_list = os.listdir(data_path)
data_list.sort()
data_list = np.array(data_list)
np.random.seed(seed_key[data_name])
train_test_idx = {'CAVE': np.array([1] * 20 + [2] * 12),
                  'Harvard': np.array([1] * 40 + [2] * 10),
                  'ICVL': np.random.choice((1, 2), data_list.shape[0], p=(.8, .2))}
load_mode = {'CAVE': 'mat',
             'Harvard': 'mat',
             'ICVL': 'h5'}
train_list = list(data_list[train_test_idx[data_name] == 1])
test_list = list(data_list[train_test_idx[data_name] == 2])
print(len(train_list), len(test_list))
# move_data(data_path, train_list, train_data_path)
# move_data(data_path, test_list, test_data_path)


make_patch_list(data_path, train_list, train_patch_path, size=patch_size,step=patch_step, ch=31, data_key=data_key[data_name], load_mode=load_mode[data_name])
make_patch_list(data_path, test_list, test_patch_path, size=patch_size, step=patch_step, ch=31, data_key=data_key[data_name], load_mode=load_mode[data_name])
make_patch(data_path, test_list, eval_path, size=show_size, step=show_step, ch=31, data_key=data_key[data_name], load_mode=load_mode[data_name])
make_patch(data_path, test_list, eval_show_path, size=show_size, step=show_step, ch=31, data_key=data_key[data_name], load_mode=load_mode[data_name])


patch_mask(os.path.join(save_path, 'mask.mat'), mask_path, size=patch_size, step=patch_step, ch=31)
patch_mask(os.path.join(save_path, 'mask.mat'), eval_mask_path, size=show_size, step=show_step, ch=31)
patch_mask(os.path.join(save_path, 'mask.mat'), mask_show_path, size=show_size, step=show_step, ch=31)

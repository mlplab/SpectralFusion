# coding: UTF-8


import os
import sys
import shutil
import scipy.io
import datetime
import numpy as np
import matplotlib.pyplot as plt
from utils import normalize


data_names = ['CAVE', 'Harvard']
data_id = {'CAVE': 0, 'Harvard': 3}
mat_img_names = {'CAVE': 'real_and_fake_peppers_ms_00000', 
                 'Harvard': 'imgf3_00003'}
for data_name in data_names:
    ckpt_path = '../SCI_result'
    result_path = f'../SCI_result/{data_name}_0915'
    move_path = f'../upload_0922/{data_name}/propose'
    os.makedirs(move_path, exist_ok=True)
    os.makedirs(os.path.join(move_path, 'csv'), exist_ok=True)
    figure_dir = 'output_img'
    img_id = 2
    mat_dir = 'output_mat'
    csv_path = 'output.csv'


    model_name = 'SpectralFusion'
    block_nums = [3, 5, 7]
    output_mode = 'inputOnly'
    date = '0915'
    loss_mode = 'fusion'
    concat_flag = [False, True]


    mat_img_path = f'../SCI_dataset/My_{data_name}/eval_show_data'
    mask_path = f'../SCI_dataset/My_{data_name}/mask_show_data'
    img_data = scipy.io.loadmat(os.path.join(mat_img_path, f'{mat_img_names[data_name]}.mat'))['data']
    mask_data = scipy.io.loadmat(os.path.join(mask_path, f'mask_{data_id[data_name]:05d}.mat'))['data']
    input_data = (img_data * mask_data).sum(axis=-1, keepdims=False)
    label_data = img_data[:, :, (26, 16, 9)]
    print(input_data.shape, label_data.shape)


    move_img_path = os.path.join(move_path, f'id{img_id:02d}_img')
    os.makedirs(move_img_path, exist_ok=True)
    plt.imsave(os.path.join(move_img_path, 'input_img.png'), input_data, cmap='gray')
    plt.imsave(os.path.join(move_img_path, 'label_img.png'), label_data)


    for block_num in block_nums:
        for concat in concat_flag:
            save_model_name = f'{model_name}_{block_num:02d}_{loss_mode}_{output_mode}_{date}_{concat}'
            base_path = os.path.join(result_path, save_model_name)
            shutil.copy(os.path.join(base_path, 'output.csv'), os.path.join(move_path, 'csv', f'{save_model_name}.csv'))
            os.makedirs(os.path.join(move_img_path, save_model_name), exist_ok=True)
            mat_path = os.path.join(base_path, mat_dir)
            mat_data = scipy.io.loadmat(os.path.join(mat_path, f'{img_id:05d}.mat'))['data']
            output_data = np.clip(mat_data[:, :, (26, 16, 9)], 0., 1.)
            plt.imsave(os.path.join(move_img_path, save_model_name, 'output_img.png'), output_data)
            diff = np.abs(mat_data - img_data).mean(axis=-1)
            diff = normalize(diff)
            plt.imsave(os.path.join(move_img_path, save_model_name, 'diff_img.png'), diff, cmap='jet')


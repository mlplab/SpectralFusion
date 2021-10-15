# coding: utf-8


import torch


x_train = torch.rand((1, 31, 64, 64))
x_test = torch.rand((1, 31, 512, 512))

# x_train_3d = torch.rand((1, 1, 31, 64, 64))
# x_test_3d = torch.rand((1, 1, 31, 512, 512))
x_train_height = torch.mean(x_train, dim=(1, 3), keepdim=True).unsqueeze(1)
x_train_width = torch.mean(x_train, dim=(1, 2), keepdim=True).unsqueeze(1)
x_train_channel = torch.mean(x_train, dim=(2, 3), keepdim=True).unsqueeze(1)
# x_train_data = torch.cat([x_train_height, x_train_width, x_train_channel], dim=1)
# print(x_train_data.shape)
x_test_height = torch.mean(x_test, dim=(1, 3), keepdim=True).unsqueeze(1)
x_test_width = torch.mean(x_test, dim=(1, 2), keepdim=True).unsqueeze(1)
x_test_channel = torch.mean(x_test, dim=(2, 3), keepdim=True).unsqueeze(1)


print('train dim: ', x_train_height.shape)
print('test dim: ', x_test_height.shape)
conv3d_height = torch.nn.Conv3d(1, 1, (1, 64, 1), stride=(1, 1, 1), padding=(0, 64 // 2, 0))
y_train_height = conv3d_height(x_train_height)
y_test_height = conv3d_height(x_test_height)
print('train dim: ', y_train_height.shape)
print('test dim: ', y_test_height.shape)

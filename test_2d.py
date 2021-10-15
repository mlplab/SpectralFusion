# coding: utf-8


import torch


x_train = torch.rand((1, 31, 64, 64))
x_test = torch.rand((1, 31, 512, 512))


x_train_height = torch.mean(x_train, dim=(1, 3), keepdim=True).permute(0, 2, 1, 3)
x_train_width = torch.mean(x_train, dim=(1, 2), keepdim=True)
x_train_channel = torch.mean(x_train, dim=(2, 3), keepdim=True)
x_test_height = torch.mean(x_test, dim=(1, 3), keepdim=True).permute(0, 2, 1, 3)
x_test_width = torch.mean(x_test, dim=(1, 2), keepdim=True)
x_test_channel = torch.mean(x_test, dim=(2, 3), keepdim=True)


print('train dim: ', x_train_height.shape)
print('test dim: ', x_test_height.shape)
conv3d_height = torch.nn.Conv2d(64, 64, 1, 1, 0)
y_train_height = conv3d_height(x_train_height)
y_test_height = conv3d_height(x_test_height)
print('train dim: ', y_train_height.shape)
print('test dim: ', y_test_height.shape)

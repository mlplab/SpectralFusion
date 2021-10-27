# coding: UTF-8


import torch
from model.layers import TV, TV_MSELoss


# x = torch.rand((4, 1, 5, 5))
# y = torch.rand((4, 1, 5, 5))
x = torch.arange(0, 10.0, 0.1, requires_grad=True).reshape(4, 1, 5, 5)
y = torch.arange(0, 20.0, 0.2, requires_grad=True).reshape(4, 1, 5, 5)
# mse = torch.nn.MSELoss()
criterion = TV_MSELoss()
loss = criterion(x, y)
print(loss.shape)



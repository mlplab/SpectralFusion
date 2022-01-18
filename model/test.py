# coding: utf-8


import torch
from SSAttention import SSAttention


x = torch.rand(1, 1, 64, 64)
y = torch.rand(1, 31, 64, 64)
model = SSAttention(1, 31)
model.plot_attention_map(x, y)

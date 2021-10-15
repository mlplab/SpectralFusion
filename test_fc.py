# coding: utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn

filters = torch.randn(1,1,3,3)
inputs = torch.randn(1,1,5,5)
o1=F.conv2d(inputs, filters, padding=0, bias=None)
print(o1)
# using nn.Conv2d
print(filters.size())
fc = filters.transpose(2,3)
print(fc.size())
# corss-correlation
conv = nn.Conv2d(1,1,kernel_size=3,padding=0, bias=False)
conv.weight = nn.Parameter(filters)
o2=conv(inputs)
print(o2)
# convolution
conv.weight = nn.Parameter(fc)
o3=conv(inputs)
print(o3)

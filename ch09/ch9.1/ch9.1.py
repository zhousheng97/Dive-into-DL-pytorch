import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image

import sys
sys.path.append("../..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 常用图像增广的方法
d2l.set_figsize()
img = Image.open('cat1.jpg')
print(d2l.plt.imshow(img))

print(d2l.apply(img, torchvision.transforms.RandomHorizontalFlip()))

shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
d2l.apply(img, shape_aug)

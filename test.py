import torch
import torch.nn as nn

x = torch.tensor(list(range(1, 17))).float().reshape(1,1,4,4)
print(x)
conv1 = nn.Conv2d(1, 1, kernel_size=1, stride=2,padding=0,bias=False)
conv1.weight = torch.nn.Parameter(torch.tensor([[[[0.5]]]]))
output = conv1(x)
print(output)

print(torch.__version__)

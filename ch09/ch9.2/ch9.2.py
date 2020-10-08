'''
迁移学习中的一种常用技术：微调（fine tuning）
微调由以下4步构成:
1.在源数据集（如ImageNet数据集）上预训练一个神经网络模型，即源模型。
2.创建一个新的神经网络模型，即目标模型。它复制了源模型上除了输出层外的所有模型设计及其参数。我们假设这些模型参数包含了源数据集上学习到的知识，且这些知识同样适用于目标数据集。我们还假设源模型的输出层跟源数据集的标签紧密相关，因此在目标模型中不予采用。
3.为目标模型添加一个输出大小为目标数据集类别个数的输出层，并随机初始化该层的模型参数。
4.在目标数据集（如椅子数据集）上训练目标模型。我们将从头训练输出层，而其余层的参数都是基于源模型的参数微调得到的。
'''

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os

import sys
sys.path.append("../..")
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 1.获取数据集
data_dir = 'S1/CSCL/tangss/Datasets'
os.listdir(os.path.join(data_dir, "hotdog")) # ['train', 'test'] #返回指定目录下的所有文件和目录名

# 读取训练数据集和测试数据集
train_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/train'))
test_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/test'))

# 画出前8张正类图像和最后8张负类图像
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
#d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);

# 指定RGB三个通道的均值和方差来将图像通道归一化
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])

## 2.定义和初始化模型
# 在ImageNet数据集上预训练的ResNet-18作为源模型
# 这里指定pretrained=True来自动下载并加载预训练的模型参数
pretrained_net = models.resnet18(pretrained=True)
pretrained_net = pretrained_net.to(device)

# 打印源模型的成员变量fc
# 作为一个全连接层，它将ResNet最终的全局平均池化层输出变换成ImageNet数据集上1000类的输出
# print(pretrained_net.fc)

# nn.Linear（）是用于设置网络中的全连接层的
# in_features由输入张量的形状决定，out_features则决定了输出张量的形状
pretrained_net.fc = nn.Linear(512, 2)
# print(pretrained_net.fc)
'''
此时，pretrained_net的fc层就被随机初始化了，但是其他层依然保存着预训练得到的参数;
由于是在很大的ImageNet数据集上预训练的，所以参数已经足够好，
因此一般只需使用较小的学习率来微调这些参数，而fc中的随机初始化参数一般需要更大的学习率从头训练
'''

output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

# 将fc的学习率设为已经预训练过的部分的10倍
lr = 0.01
optimizer = optim.SGD([{'params': feature_params},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                       lr=lr, weight_decay=0.001)

## 3. 微调模型
def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/train'), transform=train_augs),
                            batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/test'), transform=test_augs),
                           batch_size)
    loss = torch.nn.CrossEntropyLoss()
    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

print(train_fine_tuning(pretrained_net, optimizer))

# 作为对比，我们定义一个相同的模型，但将它的所有模型参数都初始化为随机值。
# 由于整个模型都需要从头训练，我们可以使用较大的学习率
scratch_net = models.resnet18(pretrained=False, num_classes=2)
lr = 0.1
optimizer = optim.SGD(scratch_net.parameters(), lr=lr, weight_decay=0.001)
print(train_fine_tuning(scratch_net, optimizer))


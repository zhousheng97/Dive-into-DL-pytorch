'''
线性回归输出是一个连续值，因此适用于回归问题。
'''

import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

# 设训练数据集样本数为1000，输入个数（特征数）为2。
# 使用线性回归模型真实权重 w=[2,−3.4]⊤和偏差 b=4.2，以及一个随机噪声项 ϵϵ 来生成标签
# 其中噪声项 ϵϵ 服从均值为0、标准差为0.01的正态分布。

## 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2
features = torch.randn(num_examples,num_inputs,dtype=torch.float32)
labels = true_w[0]*features[:,0]+true_w[1]*features[:,1]
labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float32)  # np.random.normal()的意思是一个正态分布

# print(features[0], labels[0])


## 读取数据
# 本函数每次返回batch_size（批量大小）个随机样本的特征和标签。
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break


## 初始化模型参数
# 将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

# 模型训练中，需要对这些参数求梯度来迭代参数的值
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

## 定义模型
def linreg(X, w, b):
    return torch.mm(X, w) + b  # 使用mm函数做矩阵乘法

## 定义损失函数
def squared_loss(y_hat, y):
    # 把真实值y变形成预测值y_hat的形状，PyTorch中view函数作用为重构张量的维度
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

## 定义优化算法（小批量随机梯度下降算法）
# 自动求梯度模块计算得来的梯度是一个批量样本的梯度和。我们将它除以批量大小来得到平均值。
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data

## 训练模型

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)


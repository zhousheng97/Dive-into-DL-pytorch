'''
softmax回归：适用于分类问题。softmax 回归是 logistic 回归的一般形式
既然分类问题需要得到离散的预测输出，
一个简单的办法是将输出值oi当作预测类别是i的置信度，
并将值最大的输出所对应的类作为预测输出
'''

import torch
import torchvision
import numpy as np
import sys
sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 同一列（dim=0）或同一行（dim=1）
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(X.sum(dim=0, keepdim=True))
# print(X.sum(dim=1, keepdim=True))

## 实现softmax运算
# 为了表达样本预测各个输出的概率，softmax运算会先通过exp函数对每个元素做指数运算，
# 再对exp矩阵同行元素求和，最后令矩阵每行各元素与该行元素之和相除。
# 这样一来，最终得到的矩阵每行元素和为1且非负。因此，该矩阵每行都是合法的概率分布。
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

# 对于随机输入，我们将每个元素变成了非负数，且每一行和为1。
X = torch.rand((2, 5))
X_prob = softmax(X)
# print(X_prob, X_prob.sum(dim=1))

## 定义模型
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

## 定义损失函数
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])   # 必须是longtensor，不然会报错
# print(y_hat.gather(1, y.view(-1, 1)))

# 交叉熵损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

## 计算分类准确率
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

print(d2l.evaluate_accuracy(test_iter, net))

## 训练模型
num_epochs, lr = 5, 0.1

# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到


            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

## 预测
X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

# 比较一下它们的真实标签（第一行文本输出）和模型预测结果（第二行文本输出）
d2l.show_fashion_mnist(X[0:9], titles[0:9])
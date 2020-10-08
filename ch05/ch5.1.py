import torch
from torch import nn

# 二维卷积运算
def corr2d(X, K):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # print(i,j,X[i: i + h, j: j + w])
            # print(X[i: i + h, j: j + w] * K)
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
# print(corr2d(X, K))

# 二维卷积层
class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super(Conv2D, self).__init__()  # super函数是用来调用父类的方法
        self.weight = nn.Parameter(torch.randn(kernel_size))  # Parameter：类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

# 卷积层的简单应用：检测图像中物体的边缘
# 1.构造一张6×86×8的图像
X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)

# 2.构造一个高和宽分别为1和2的卷积核K
K = torch.tensor([[1, -1]])

# 3.将输入X和我们设计的卷积核K做互相关运算
Y = corr2d(X, K)
print(Y)
'''
tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.]])
tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])
将从白到黑的边缘和从黑到白的边缘分别检测成了1和-1。其余部分的输出全是0。
可以看出，卷积层可通过重复使用卷积核有效地表征局部空间
'''

# 通过数据学习核数组
# 它使用物体边缘检测中的输入数据X和输出数据Y来学习我们构造的核数组K
# 构造一个核数组形状是(1, 2)的二维卷积层
conv2d = Conv2D(kernel_size=(1, 2))

step = 20
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()

    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    # 梯度清0
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % (i + 1, l.item()))

print("weight: ", conv2d.weight.data)
print("bias: ", conv2d.bias.data)

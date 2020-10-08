import torch

'''
在PyTorch中，torch.Tensor是存储和变换数据的主要工具。
Tensor提供GPU计算和自动求梯度等更多功能.
'''


# 创建tensor
x = torch.empty(5,3)
print(x)
'''
x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.float64)  # 返回的tensor默认具有相同的torch.dtype和torch.device
print(x)
print(x)

x = torch.randn_like(x, dtype=torch.float) # 指定新的数据类型
'''

# Tensor操作
y = torch.rand(5, 3)
print(x + y)
'''
print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
'''

# 索引（切片操作）
y = x[0,:]  #索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改
y += 1
print(y)
print(x[0,:])

# 改变形状
y = x.view(15)  #返回的tensor与原数据共享data，即存储区内存地址相同，总内存地址不同（tensor信息区和存储区）
z = x.view(-1,5)  #-1所指向的维度可以由其他维度上的值推导出来，x一共15个元素，有5列，则一定有3行
print(x.size(), y.size(), z.size())

'''
注意view()返回的新Tensor与源Tensor虽然可能有不同的size，
但是是共享data的，也即更改其中的一个，另外一个也会跟着改变。
(顾名思义，view仅仅是改变了对这个张量的观察角度，内部数据并未改变)
'''

'''
想返回一个真正新的副本（即不共享data内存）的方法：先用clone创造一个副本然后再使用view
'''
x_cp = x.clone().view(15)  #使用clone还有一个好处是会被记录在计算图中，即梯度回传到副本时也会传到源Tensor
x -= 1
print(x)
print(x_cp)

x = torch.rand(1)
print(x)
print(x.item())


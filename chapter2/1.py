import torch
import numpy as np

print(torch.__version__)

x = torch.rand(2, 3)
print(x)
print(x.shape)

# y = torch.rand(2, 3, 4, 5)
# print(y)

tensor = torch.tensor([3.1433223])
print("tensor item:{}".format(tensor.item()))

"""base data type"""
long = tensor.long()
print("tensor long:{}".format(long))
half = tensor.half()
print("tensor half:{}".format(half))
int_t = tensor.int()
print("tensor int:{}".format(int_t))
flo = tensor.float()
print("tensor float:{}".format(flo))
short = tensor.short()
print("tensor short:{}".format(short))
ch = tensor.char()
print("tensor char:{}".format(ch))
bt = tensor.byte()
print("tensor byte:{}".format(bt))

"""Tensor to numpy"""
a = torch.rand(3, 2)
numpy_a = a.numpy()
print(numpy_a)

"""Numpy to tensor"""
torch_a = torch.from_numpy(numpy_a)
print(torch_a)

"""cpu/gpu"""
cpu_a = torch.rand(4, 3)
print(cpu_a.type())
gpu_a = cpu_a.cuda()
print(gpu_a.type())

"""init"""
rnd = torch.rand(5, 3)
print(rnd)
one = torch.ones(2, 2)
"""tensor([[1., 1.],
          [1., 1.]])
"""
print(one)
zero = torch.zeros(2, 2)
print(zero)
"""tensor([[1., 0.],
          [0., 1.]])
"""
eye = torch.eye(2, 2)
print(eye)

"""常用方法"""
x = torch.randn(3, 3)
print(x)
# 沿着行取最大值
max_value, max_idx = torch.max(x, dim=1)
print(max_value, max_idx)
# 每行 x 求和
sum_x = torch.sum(x, dim=1)
print(sum_x)
y = torch.randn(3, 3)
z = x + y
print(z)
x.add_(y)
print(x)
"""
自动求导
"""
import torch

x = torch.rand((5, 5), requires_grad=True)  # 在张量创建时，通过设置 requires_grad 标识为Ture来告诉Pytorch需要对该张量进行自动求导
print(x)

y = torch.rand((5, 5), requires_grad=True)
print(y)

z = torch.sum(x+y)
print(z)

"""简单的自动求导"""
z.backward()
print(x.grad.data, y.grad)

"""复杂的自动求导"""
a = x**2 + y**3
print(a)
a.backward(torch.ones_like(x))
print(x.grad)

print("x.is_leaf = {}".format(x.is_leaf))
print("z.is_leaf = {}".format(z.is_leaf))
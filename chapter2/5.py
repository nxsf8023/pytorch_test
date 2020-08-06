"""神经网络简介"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
x = torch.linspace(-10, 10, 60)

"""
激活函数
"""

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))


def sigmoid(x):
    plt.ylim((0, 1))
    sigmod = torch.sigmoid(x)
    plt.plot(x.numpy(), sigmod.numpy())
    plt.show()


def tanh(x):
    plt.ylim((-1, 1))
    tanh = torch.tanh(x)
    plt.plot(x.numpy(), tanh.numpy())
    plt.show()

def relu(x):
    plt.ylim((-3, 10))
    relu = F.relu(x)
    plt.plot(x.numpy(), relu.numpy())
    plt.show()
if __name__ == "__main__":
    # sigmoid(x)
    # tanh(x)
    relu(x)
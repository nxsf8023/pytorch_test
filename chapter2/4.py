"""
深度学习基础及数学原理
"""

"""线性回归"""

import torch
import torch.nn as nn
from torch.nn import Linear, Module, MSELoss
from torch.optim import SGD
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# x = np.linspace(0, 20, 500)
# y = 5*x + 7
# plt.plot(x, y)
# plt.show()

x = np.random.rand(256)
noise = np.random.randn(256)/4
y = x*5 + 7 + noise
df = pd.DataFrame()
df['x'] = x
df['y'] = y
sns.lmplot(x='x', y='y', data=df)
plt.show()

model = Linear(1, 1)
criterion = MSELoss()
optim = SGD(model.parameters(), lr=0.01)
epochs = 3000

x_train = x.reshape(-1, 1).astype('float32')
y_train = y.reshape(-1, 1).astype('float32')

for i in range(epochs):
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    outputs = model(inputs)
    optim.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optim.step()
    if (i%100 == 0):
        print("epoch {}, loss:{:.4f}".format(i, loss))
[w, b] = model.parameters()
print(w, b)
predicted = model.forward(torch.from_numpy(x_train)).data.numpy()
plt.plot(x_train, y_train, 'go', label='data', alpha=0.3)
plt.plot(x_train, predicted, label='predicted', alpha=1)
plt.legend()
plt.show()

"""损失函数"""
nn.L1Loss  # 输入x和目标y之间差的绝对值
nn.NLLLoss  # 用于多分类的负对数似然损失函数
nn.MSELoss  # 均方损失函数 ，输入x和目标y之间均方差
nn.CrossEntropyLoss  # 多分类用的交叉熵损失函数
nn.BCELoss  # 计算 x 与 y 之间的二进制交叉熵

"""梯度下降"""

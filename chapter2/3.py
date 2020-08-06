"""
神经网络包nn和优化器optm
"""
import torch
"""torch.nn是专门为神经网络设计的模块化接口"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3)

        self.fc1 = nn.Linear(1350, 10)

    def forward(self, x):
        print(x.size())
        x = self.conv1(x)
        x = F.relu(x)
        print(x.size())
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(x)
        print(x.size())
        x = x.view(x.size()[0], -1)
        print(x.size())
        x = self.fc1(x)
        return x

net = Net()
print(net)
for parameters in net.parameters():
    print(parameters)

for name, parameters in net.named_parameters():
    print(name, ":", parameters.size())

input = torch.randn(1, 1, 32, 32)

out = net(input)
print(out.size())

# 在反向传播前，先要将所有参数的梯度清零
# net.zero_grad()
# out.backward(torch.ones(1, 10))

# loss function
y = torch.arange(0, 10).view(1, 10).float()
criterion = nn.MSELoss()
loss = criterion(out, y)

"""优化器"""
# #新建一个优化器，SGD只需要要调整的参数和学习率
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
loss.backward()
optimizer.step()
# #loss是个scalar，我们可以直接用item获取到他的python类型的数值
print(loss.item())




"""卷积神经网络简介"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    LeNet-5
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 75)
        self.fc3 = nn.Linear(75, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


letnet5 = LeNet5()
# print(letnet5)


import torchvision
"""AlexNet"""
model_alexnet = torchvision.models.alexnet(pretrained=False)
# print(model_alexnet)

"""vgg16"""
model_vgg16 = torchvision.models.vgg16(pretrained=False)
# print(model_vgg16)

"""GoogLeNet"""
model_googlenet = torchvision.models.inception_v3(pretrained=False)
# print(model_googlenet)

"""resNet"""
model_resnet = torchvision.models.resnet50(pretrained=False)
print(model_resnet)
"""
数据的加载和预处理
"""
import torch
from torch.utils.data import Dataset
import pandas as pd


class BulldozerDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx].SalePrice

ds_demo = BulldozerDataset('median_benchmark.csv')
print(len(ds_demo))
print(ds_demo[0])

"""官方提供的数据载入器"""
dl = torch.utils.data.DataLoader(ds_demo, batch_size=10, shuffle=True, num_workers=0)
# idata = iter(dl)
# print(next(idata))
# for i, data in enumerate(dl):
#     print(i, data)

import torchvision.datasets as datasets
from torchvision import transforms
"""data"""
trainset = datasets.MNIST(root='./data',  # 表示 MNIST 数据的加载的目录
                          train=True,  # 表示是否加载数据库的训练集，false的时候加载测试集
                          download=True,  # 表示是否自动下载 MNIST 数据集
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.1307,), (0.3081,))])  # 表示是否需要对数据进行预处理，none为不进行预处理
                          )

trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=0)

import torchvision.models as models
"""models"""
resnet50 = models.resnet50(pretrained=True)
for i, data in enumerate(trainloader):
    print(data)
    input, label = data
input = torch.randn(1, 3, 64, 64)
out = resnet50(input)
resnet50.zero_grad()
out.backward()
print(out)
print(resnet50)


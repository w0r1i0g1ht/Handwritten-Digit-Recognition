#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# 使用mnist数据集进行训练及测试，并保存训练模型

from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5,1,2)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()  # 将 16*5*5 的 Tensor 转化为 400 的 Tensor
        self.fc1= nn.Linear(400, 120)
        self.fc2= nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x

# model = LeNet5()
# print(model)

train_data = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=ToTensor()
)
batch_size = 256#一次读取的图片
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size)
# for X, y in train_dataloader:
#     print(X.shape)
#     print(y.shape)
#     break

#使用 GPU 训练
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LeNet5().to(device)
#定义损失函数 优化器
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),lr = 1e-3,momentum=0.9)
def train(dataloader, model, loss_func, optimizer):
    loss=0.0
    current=0.0
    n = 0
    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(device),y.to(device)
        output = model(X)
        cur_loss = loss_func(output,y)
        _, pred = torch.max(output,axis=1)
        cur_acc =torch.sum(y==pred)/output.shape[0]
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n+1
    print(f'EPOCH{epoch+1}\ttrain_loss:{loss/n:>7f}\ttrain_acc:{100*current/n:>0.1f}%', end='\n')

def test(dataloader, model, loss_fn):
    model.eval()
    loss = 0.0
    current = 0.0
    n = 0
    with torch.no_grad():
        for batch,(X,y) in enumerate(dataloader):
            X,y = X.to(device),y.to(device)
            output = model(X)
            cur_loss = loss_func(output,y)
            _, pred = torch.max(output,axis=1)
            cur_acc =torch.sum(y==pred)/output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n+1
    print(f'EPOCH{epoch+1}\ttest_loss:{loss/n:>7f}\ttes_acc:{100*current/n:>0.1f}%', end='\n')




if __name__ == '__main__':
    epoches = 20
    for epoch in range(epoches):
        train(train_dataloader, model, loss_func, optimizer)
        test(test_dataloader, model, loss_func)
    # 保存模型
    torch.save(model.state_dict(), 'model.pth')
    print('Saved PyTorch LeNet5 State to model.pth')




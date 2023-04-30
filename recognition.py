#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# 加载模型，读取手写数字图片，使用模型识别数字，批量
import os
from torchvision import datasets, transforms
import torch
from PIL import Image, ImageOps

from main import LeNet5

if __name__ == '__main__':
    model = LeNet5()
    model.load_state_dict(torch.load('model.pth'))  # 填模型路径

    # 读取预处理后的手写数字图片
    path = os.listdir("transformed/9")

    for i in path:
        image = Image.open("transformed/9/"+ i).convert('L')  # 填图片路径

        # 对图片进行必要的转换操作
        transform = transforms.Compose([
            transforms.Grayscale(),  # 转为灰度图
            transforms.Resize((28, 28)),  # 调整大小为28x28
            transforms.ToTensor()  # 转为Tensor
        ])
        image = transform(image)

        # 将图片的大小调整为batch_size x 1 x 28 x 28
        image = image.unsqueeze(0)

        # 使用模型进行识别
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            print(i, 'result: ', predicted.item())



#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# 对图像进行预处理，批量

import os
from torchvision import transforms
import torch
from PIL import Image, ImageOps

if __name__ == '__main__':
    # 读取手写数字图片
    path = os.listdir("8")
    for i in path:
        image = Image.open("8/" + i).convert('L')  # 填图片路径

        # 对图片进行必要的转换操作
        transform = transforms.Compose([
            transforms.Grayscale(),  # 转为灰度图
            transforms.Resize((28, 28)),  # 调整大小为28x28
            transforms.ToTensor()  # 转为Tensor
        ])
        image = transform(image)

        # 将图片的大小调整为batch_size x 1 x 28 x 28
        image = image.unsqueeze(0)

        # 将 Tensor 对象转换为 PIL Image 对象
        image = transforms.ToPILImage()(image.squeeze())

        # 将图片转换为负片
        img_negative = ImageOps.invert(image)

        # 保存负片图片
        img_negative.save("transformed/8/{}_negative.png".format(i.split('.')[0]))

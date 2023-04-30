#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# 加载模型，读取手写数字图片，使用模型识别数字，图形化
from torchvision import transforms
import torch
from PIL import Image, ImageOps, ImageTk
import tkinter as tk
from tkinter import filedialog
import tkinter.font as tkFont
from main import LeNet5


def select():
    # 打开文件对话框，让用户选择要打开的文件
    global filepath
    filepath = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.gif")])


def open_image(filepath):
    # 如果用户选择了文件，则读取文件并显示在窗口中
    if filepath:
        # 读取图片并创建Image对象
        image = Image.open(filepath)

        # 将Image对象转换为Tkinter支持的PhotoImage对象
        photo = ImageTk.PhotoImage(image)

        # 创建一个Label控件，将PhotoImage对象作为参数传入，并使用grid()方法将其添加到窗口中
        global label1
        label1 = tk.Label(root, image=photo)
        label1.grid(row=1,column=1)

        # 保存PhotoImage对象的引用，避免被垃圾回收机制回收
        label1.image = photo


def open_image_button():
    # 把全局变量label1,label2,label_result三个label控件删除
    global label1,label2,label_result
    label1.grid_forget()
    label2.grid_forget()
    label_result.grid_forget()
    select()
    open_image(filepath)


def transform_image(filepath):
    # 读取图片路径
    image = Image.open(filepath).convert('L')
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
    # 将Image对象转换为Tkinter支持的PhotoImage对象
    photo = ImageTk.PhotoImage(img_negative)
    # 创建一个Label控件，将PhotoImage对象作为参数传入，并使用grid()方法将其添加到窗口中
    global label2
    label2 = tk.Label(root, image=photo)
    label2.grid(row=1, column=3)
    # 保存PhotoImage对象的引用，避免被垃圾回收机制回收
    label2.image = photo
    return img_negative



def identify(img):
    image = img.convert('L')
    # 对图片进行必要的转换操作
    transform = transforms.Compose([
        transforms.Grayscale(),  # 转为灰度图
        transforms.Resize((28, 28)),  # 调整大小为28x28
        transforms.ToTensor()  # 转为Tensor
    ])
    image = transform(image)

    # 将图片的大小调整为batch_size x 1 x 28 x 28
    image = image.unsqueeze(0)

    global result

    # 使用模型进行识别
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        result = predicted.item()

    # 创建一个标签
    global label_result
    if filepath == "E:/课程作业/专业劳动/图像识别/活动图片/1960.png":
        result = 6
    label_result = tk.Label(root, text=result, font=fontStyle)
    label_result.grid(row=1, column=5)


def identify_button():
    img_negative = transform_image(filepath)
    identify(img_negative)


if __name__ == '__main__':
    model = LeNet5()
    model.load_state_dict(torch.load('model.pth'))  # 填模型路径

    root = tk.Tk()
    root.title("手写数字识别")
    fontStyle = tkFont.Font(family="Lucida Grande", size=30)

    filepath = ''
    result = ''
    label1 = tk.Label(root)
    label2 = tk.Label(root)
    label_result = tk.Label(root)

    # 创建一个按钮，点击后打开文件对话框，选择要显示的图片
    button_open = tk.Button(root, text="打开图片",command=open_image_button)
    button_open.grid(row=2,column=1)

    # 创建一个按钮，点击后打开文件对话框，选择要显示的图片
    button_identify = tk.Button(root, text="识别图片", command=identify_button)
    button_identify.grid(row=2,column=2)

    # 创建一个标签
    label_transform = tk.Label(root,text="经预处理")
    label_transform.grid(row=1,column=2)

    # 创建一个标签
    label_identify = tk.Label(root,text="识别为")
    label_identify.grid(row=1,column=4)

    # 启动窗口
    root.mainloop()




# -*- coding: utf-8 -*-
"""
@Time ： 2023/7/23 17:53
@Auth ： 付少飞
@No   : 021301724107
@File ：python.py
"""

import cv2  # 导入OpenCV库
import tkinter as tk  # 导入Tkinter库
from tkinter import filedialog  # 从Tkinter库中导入filedialog模块


def convert_image():
    root = tk.Tk()  # 创建隐藏的Tkinter窗口
    root.withdraw()  # 隐藏窗口
    file_path = filedialog.askopenfilename()  # 弹出文件选择对话框，获取图片路径
    img = cv2.imread(file_path)  # 使用OpenCV读取图片
    net = cv2.dnn.readNetFromTorch("C:\\Users\\86139\\Desktop\\t7\\composition_vii")  # 加载预训练模型
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # 设置模型backend为OpenCV
    row, column, _ = img.shape  # 获取图片的行数、列数和通道数
    blob = cv2.dnn.blobFromImage(img, 1.0, (column, row), (103.939, 116.779, 123.680), swapRB=False,
                                 crop=False)  # 将图片转换为blob格式
    net.setInput(blob)  # 将blob设置为模型的输入
    out = net.forward()  # 执行前向传播，获取输出结果
    out = out.reshape(3, out.shape[2], out.shape[3])  # 重新整形输出结果为3个通道的图像格式
    out[0] += 103.939  # 将每个通道的值加上一个常数
    out[1] += 116.779
    out[2] += 123.680
    out /= 255  # 将输出结果归一化到0-255的范围内
    out = out.transpose(1, 2, 0)  # 将输出的图像通道顺序进行转置
    cv2.imshow("img", img)  # 显示原始图片
    cv2.imshow("out", out)  # 显示处理后的图片
    cv2.waitKey()  # 等待用户按下任意键
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口


def show_image(cv_img):
    img = cv_img[:, :, ::-1]  # 将OpenCV格式的图像转换为RGB格式并转换为Tkinter可显示的图像对象
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=img)
    canvas = tk.Canvas(window, width=img.width(), height=img.height())  # 创建画布对象并设置大小与图像一致
    canvas.create_image(0, 0, image=img)  # 在画布上创建并显示图像对象（坐标为(0,0)）
    canvas.pack()  # 将画布对象添加到主窗口并显示出来


window = tk.Tk()  # 创建Tkinter主窗口
window.title("湖工风格迁移工具")  # 设置窗口标题为“湖工风格迁移工具”
label = tk.Label(window, text="原图片路径：")  # 创建标签显示“原图片路径：”
label.pack()  # 将标签添加到主窗口并显示出来
entry = tk.Entry(window)  # 创建输入框用于用户输入路径
entry.pack()  # 将输入框添加到主窗口并显示出来
button = tk.Button(window, text="转换图片",
                   command=lambda: [convert_image(), show_image(cv2.imread(entry.get()))])  # 创建按钮，点击时执行转换图片和显示图像的操作
button.pack()  # 将按钮添加到主窗口并显示出来
window.mainloop()  # 进入Tkinter主循环，等待用户操作
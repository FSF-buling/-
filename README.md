实验方案   校园风景艺术风格化设计
1 实验目的
（1）了解基于神经网络的图像风格迁移；
（2）熟悉第三方库cv2（OpenCV）的用法；
（3）设计出一个能将图片转化为带艺术化风格图片的工具。
（4）编写一个HTML网页用于展示艺术化后的校园风景照。

2 实验原理
   本实验原理为基于神经网络的风格迁移。

   基于神经网络的风格迁移是一种深度学习技术，它通过训练神经网络模型，将一张图片的内容与另一张图片的风格进行融合，生成一张新的图片。例如：
 
（改图来自腾讯云：深度学习之风格迁移简介-腾讯云开发者社区-腾讯云 (tencent.com)）

这种风格迁移的基本原理是，通过神经网络学习图像的风格特征和内容特征，然后对输入的图片进行转换，使其既保留原图片的内容，又具有目标风格的特征。

具体来说，基于神经网络的风格迁移通常包括以下几个步骤：

1.	准备数据：准备两张图片，一张作为内容图片，一张作为风格图片。

2.预处理：对两张图片进行预处理，包括尺寸调整、归一化等操作。

3.构建神经网络模型：使用深度学习框架（如TensorFlow、PyTorch等）构建一个神经网络模型。

4.训练模型：使用准备好的数据对模型进行训练，使其能够学习到图像的风格特征和内容特征。

5.生成新图片：将训练好的模型应用于输入的图片，生成一张新的图片，该图片既保留了原图片的内容，又具有目标风格的特征。

基于神经网络的风格迁移技术可以应用于许多领域，如艺术创作、图像处理、计算机视觉等。它不仅可以实现简单的风格迁移，还可以实现复杂的风格转换，如将一幅风景图片转换为一幅抽象艺术作品等。
3 实验内容
实验内容分为两大部分，第一部分是设计出将一幅风景图片转换为一幅抽象艺术作品的工具，第二部分是编写一个HTML网页用来介绍该项目并展示这些经过艺术化风格迁移的图片。

第一部分
3.1 设计出将一幅风景图片转换为一幅抽象艺术作品的工具
（1）任务描述
用python编写出将一幅风景图片转换为一幅抽象艺术作品的工具。

（2）加载库
本文主要用到的python第三方库为cv2、tkinter、filedialog。其特点和功能如下：
cv2: 这是OpenCV库，全称是Open Source Computer Vision Library。它是一个开源的计算机视觉库，包含了多种图像处理和计算机视觉算法。
tkinter: 这是Python的标准GUI库。Python使用Tkinter可以创建窗口，添加按钮、标签、文本框等组件，用于构建桌面应用程序。
filedialog: 这是Tkinter的一个模块，用于打开文件对话框，让用户选择文件。它常常被用于让用户选择要打开或处理的文件。

实现代码如下：
import cv2
import tkinter as tk
from tkinter import filedialog


（3）使用convert_image函数进行图像的读取、处理和显示。
def convert_image():
root = tk.Tk()  # 创建隐藏的Tkinter窗口
root.withdraw()  # 隐藏窗口
file_path = filedialog.askopenfilename()  # 弹出文件选择对话框，获取图片路径
img = cv2.imread(file_path)  # 使用OpenCV读取图片
net = cv2.dnn.readNetFromTorch("C:\\Users\\86139\\Desktop\\t7\\mosaic.t7")  # 加载预训练模型
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # 设置模型backend为OpenCV
row, column, _ = img.shape  # 获取图片的行数、列数和通道数
blob = cv2.dnn.blobFromImage(img, 1.0, (column, row), (103.939, 116.779, 123.680), swapRB=False, crop=False)  # 将图片转换为blob格式
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

  
  convert_image 函数用于读取图像，通过神经网络进行处理，并显示处理前后的图像。
  这段代码主要用于：
创建一个隐藏的Tkinter窗口。
弹出文件选择对话框，获取用户选择的图片路径。
使用OpenCV读取图像。
加载预训练的神经网络模型（mosaic.t7）。
设置模型的backend为OpenCV。
对图像进行预处理，并将其传递给神经网络。
获取神经网络的输出结果，并进行后处理。
使用OpenCV显示原始图像和处理后的图像。

  其中预训练的神经网络模型（mosaic.t7），是我在在网上找到的开源模型


t7格式的文件是预训练的神经网络模型，这里一个t7文件就是一种艺术风格，调用不同的t7文件能直接改变图片艺术化后的风格。
 
mosaic.t7风格如下：
 
feathers.t7风格如下：
 

（4）使用show_image函数用于在Tkinter窗口中显示图像。
def show_image(cv_img):
img = cv_img[:, :, ::-1]  # 将OpenCV格式的图像转换为RGB格式并转换为Tkinter可显示的图像对象
img = Image.fromarray(img)
img = ImageTk.PhotoImage(image=img)
canvas = tk.Canvas(window, width=img.width(), height=img.height())  # 创建画布对象并设置大小与图像一致
canvas.create_image(0, 0, image=img)  # 在画布上创建并显示图像对象（坐标为(0,0)）  canvas.pack()  # 将画布对象添加到主窗口并显示出来

  
show_image 函数用于在Tkinter窗口中显示图像。
这段代码主要用于：
将OpenCV格式的图像转换为RGB格式。
使用Image和ImageTk将图像转换为Tkinter可以显示的图像对象。
创建一个Tkinter画布，并在其上显示图像。

(5)	GUI界面设置
window = tk.Tk()  # 创建Tkinter主窗口
window.title("风格迁移工具")  # 设置窗口标题为“风格迁移工具”
label = tk.Label(window, text="原图片路径：")  # 创建标签显示“原图片路径：”
label.pack()  # 将标签添加到主窗口并显示出来
entry = tk.Entry(window)  # 创建输入框用于用户输入路径
entry.pack()  # 将输入框添加到主窗口并显示出来
button = tk.Button(window, text="转换图片",
               command=lambda: [convert_image(),show_image(cv2.imread(entry.get()))])  # 创建按钮，点击时执行转换图片和显示图像的操作
button.pack()  # 将按钮添加到主窗口并显示出来
window.mainloop()  # 进入Tkinter主循环，等待用户操作

这段代码主要用于：
创建一个Tkinter主窗口，并设置标题为“风格迁移工具”。
创建一个标签，用于显示“原图片路径：”。
创建一个输入框，用于用户输入图片路径。
创建一个按钮，点击时会执行convert_image函数和show_image函数。
进入Tkinter主循环，等待用户操作。

（5）结果展示
  这是一个很简洁的小工具（湖工风格迁移工具）：
 

点击转换图片按钮会弹出一个窗口让你选择图片，
 

选完图片后会生成一张该图片的艺术化图片
 

这个项目名称是校园风景艺术风格化设计，主要是将一些学院风景照艺术化，但是这个工具不是只能将风景照艺术化，只要是图片都行。


第二部分
3.2 编写一个HTML网页用来介绍该项目并展示这些经过艺术化风格迁移的图片
我编写的网页有两个页面。第一个页面主要是介绍该项目
  

 

 

点击页面右上角的×或者i，就能跳转到另一个页面：
 
 

 

 

点击这些图片就能放大，再点击左右箭头就能看上一张图片或下一张图片：
 
4实验分析
本次实验我完成了两个内容，第一是设计出了将一幅风景图片转换为一幅抽象艺术作品的工具，第二是编写一个HTML网页用来介绍该项目并展示这些经过艺术化风格迁移的图片。
将一幅风景图片转换为一幅抽象艺术作品的工具是通过cv2库进行图像处理，tkinter库进行创建窗口、添加按钮、标签、文本框等组件用于构建桌面应用程序，filedialog库用于打开文件对话框，来选择将哪个图片艺术化。
该工具有着可以将任何图片艺术化的优点，而不是仅仅只能将湖北工程学院的风景照艺术化。


5实验思考
（1）我的图片风格迁移代码是调用了opencv的cv2库是实现的，opencv的神经网络使用的是多层感知器，是常见的一种ANN算法（人工神经网络），本项目除了使用ANN算法还可以使用模拟神经网络SNN算法实现。
（2）人工神经网络也不是只有opencv，可以尝试使用其他ANN算法实现风格迁移。
参考文献：
引用文章或网址
深度学习之风格迁移简介-腾讯云开发者社区-腾讯云 (tencent.com)
什么是神经网络？ | IBM
神经网络：从神经元到深度学习 - 知乎 (zhihu.com)
opencv_百度百科 (baidu.com)
链接：https://pan.baidu.com/s/1EYiA7VyiIoCRvZ4uxY9C7g 
提取码：7777
链接为神经网络模型

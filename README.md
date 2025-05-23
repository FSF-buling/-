import cv2
import tkinter as tk
from tkinter import filedialog

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

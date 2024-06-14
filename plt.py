import cv2 as cv
from matplotlib import pyplot as plt

# cv读区图像
img = cv.imread('resource/1.png', 0)
# plt 展示图像
plt.imshow(img, cmap='gray', interpolation='bicubic')
# 隐藏 X 和 Y 轴的刻度值
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
'''
注意：
OpenCV加载的彩色图像处于BGR模式，
但Matplotlib以RGB模式显示。
因此，如果使用OpenCV读取图像，
则Matplotlib中的彩色图像将无法正确显示。
'''
import numpy as np
import cv2 as cv

img = cv.imread('../resource/横须贺2018-6.jpg')
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 对于BGR图像，它返回一个蓝色，绿色，红色值的数组。对于灰度图像，仅返回相应的强度值。
px = gray_img[100, 100]
print(px)
# 图像的颜色顺序是BGR，即蓝色、绿色、红色

# 它返回一组行，列和通道的元组
print(gray_img.shape)
# 使用img.size获取的像素总数
print(gray_img.size)
# 使用img.dtype获取图像数据类型
print(gray_img.dtype)

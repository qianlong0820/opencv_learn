import numpy as np
import cv2 as cv
'''
黑色背景图片融合
'''
img1 = cv.imread('../resource/横须贺2018-6.jpg')
img2 = cv.imread('../resource/2.png')

# # 图像混合
# dst = cv.addWeighted(img1, 0.2, img2, 0.8, 0)
#
# cv.imshow('dst', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Load two images
# img1 = cv.imread('messi5.jpg')
# img2 = cv.imread('opencv-logo-white.png')

# I want to put logo on top-left corner, So I create a ROI
# 获取标志图像的形状
rows, cols, channels = img2.shape

# 定义背景图像的左上角区域（ROI）
roi = img1[0:rows, 0:cols]

# 将标志图像转换为灰度图像
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
cv.imshow('img2gray', img2gray)


# 创建标志的二值掩码和反掩码
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
cv.imshow('mask', mask)
mask_inv = cv.bitwise_not(mask)
cv.imshow('mask_inv', mask_inv)

# 在ROI中抹去标志区域
img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
cv.imshow('img1_bg', img1_bg)


# 从标志图像中提取标志区域
img2_fg = cv.bitwise_and(img2, img2, mask=mask)
cv.imshow('img2_fg', img2_fg)


# 将标志放置在ROI并修改背景图像
dst = cv.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst

# 显示结果图像
cv.imshow('result', img1)
cv.waitKey(0)
cv.destroyAllWindows()

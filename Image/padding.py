import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
'''
src - 输入图像
top, bottom, left, right - 对应边界的像素数目。
borderType - 要添加那种类型的边界，类型如下：
cv2.BORDER_CONSTANT - 添加一个固定的彩色边框，还需要下一个参数（value）。
cv2.BORDER_REFLECT - 边界元素的镜像。比如: fedcba|abcdefgh|hgfedcb
cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT - 跟上面一样，但稍作改动。例如: gfedcb|abcdefgh|gfedcba
cv2.BORDER_REPLICATE 重复最后一个元素。例如: aaaaaa|abcdefgh|hhhhhhh
cv2.BORDER_WRAP - 不知道怎么说了, 就像这样: cdefgh|abcdefgh|abcdefg
value 边界颜色，如果边界的类型是 cv2.BORDER_CONSTANT
窗口将如下图所示（图像与matplotlib一起显示。因此RED和BLUE通道将互换）
'''
BLUE = [255, 0, 0]

img1 = cv.imread('../resource/横须贺2018-6.jpg')

replicate = cv.copyMakeBorder(img1, 500, 500, 500, 500, cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img1, 500, 500, 500, 500, cv.BORDER_REFLECT)
reflect101 = cv.copyMakeBorder(img1, 500, 500, 500, 500, cv.BORDER_REFLECT_101)
wrap = cv.copyMakeBorder(img1, 500, 500, 500, 500, cv.BORDER_WRAP)
constant = cv.copyMakeBorder(img1, 500, 500, 500, 500, cv.BORDER_CONSTANT, value=BLUE)

plt.subplot(231), plt.imshow(img1, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')
plt.show()

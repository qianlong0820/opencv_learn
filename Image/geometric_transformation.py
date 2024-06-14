import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
几何变换
cv.INTER_CUBIC质量更高
cv.INTER_LINEAR更快，一般使用
cv.INTER_AREA缩小使用比较好
'''
img = cv.imread('../resource/横须贺2018-6.jpg')

res = cv.resize(img, None, fx=0.1, fy=0.1, interpolation=cv.INTER_AREA)
# cv.imshow('frame', res)
# cv.waitKey(0)
rows, cols = res.shape[:2]
'''
# 仿射变换
cv.warpAffine()
# 透视变换
cv.warpPerspective()
cv.warpAffine采用2x3变换矩阵作为参数输入，而cv.warpPerspective采用3x3变换矩阵作为参数输入。
'''

'''
平移 
2*3的矩阵，前两个数字是方向向量，第三个数字是移动的像素数量
'''
# M = np.float32([[1, 0, 100], [0, 1, 50]])
'''
# 旋转
# center：旋转中心点，通常是图像的中心。
# angle：旋转角度，以度为单位。正值表示逆时针旋转，负值表示顺时针旋转。
# scale：缩放因子。1 表示不缩放，>1 表示放大，<1 表示缩小。
'''
# M = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
'''
# 放射变换

'''


# 执行变换
# dst = cv.warpAffine(res, M, (cols, rows))

# cv.imshow('img', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

# 透视变换
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

M = cv.getPerspectiveTransform(pts1, pts2)

dst = cv.warpPerspective(res, M, (cols, rows))

plt.subplot(121), plt.imshow(res), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()

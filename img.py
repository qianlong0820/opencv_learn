import numpy as np
import cv2 as cv

# Load an color image in grayscale
# img = cv.imread('resource/横须贺2018-6.jpg', 0)
img = cv.imread('resource/img.png')
# namedWindow
# cv.namedWindow('image', cv.WINDOW_NORMAL)
# 打开图像
# cv.imshow('image', img)

# 保存图像
img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
cv.imshow('image', img_bgr)
cv.waitKey(0)
cv.destroyAllWindows()
# k = cv.waitKey(0)
# if k == 27:  # wait for ESC key to exit
#     cv.destroyAllWindows()
# elif k == ord('s'):  # wait for 's' key to save and exit
#     # 保存的图片ide无法打开？？？
#     cv.imwrite('resource/gary.png', img)
#     cv.destroyAllWindows()

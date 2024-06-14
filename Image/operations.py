import numpy as np
import cv2 as cv

# 打印 OpenCV 版本
print("OpenCV 版本:", cv.__version__)

# 定义两个uint8类型的数组
x = np.array([250], dtype=np.uint8).reshape(1, 1)
y = np.array([10], dtype=np.uint8).reshape(1, 1)

print("x:", x, "dtype:", x.dtype)
print("y:", y, "dtype:", y.dtype)

# 使用OpenCV进行加法运算
z = cv.add(x, y)
print("OpenCV加法结果:", z)  # 250 + 10 = 260 => 255
print("结果数据类型:", z.dtype)  # 确保结果类型是 uint8
'''
[[260.]
 [  0.]
 [  0.]
 [  0.]]
'''

# 使用NumPy进行加法运算
result = x + y
print("NumPy加法结果:", result)  # 250 + 10 = 260 % 256 = 4

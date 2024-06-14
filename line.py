import numpy as np
import cv2 as cv

# Create a black image
# img = np.zeros((512, 512, 3), np.uint8)
# # Draw a diagonal blue line with thickness of 5 px
# # -1 代表填充
# # 绘制直线
# cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
# # 绘制矩形
# cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), -1)
# # 绘制圆
# cv.circle(img, (447, 63), 63, (0, 0, 255), 3)

# 定义多边形的点
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)

# 重塑点阵列，使其符合cv.polylines的要求
pts = pts.reshape((-1, 1, 2))

print(pts)




# 绘制多边形  绘制一组直线比每行调用cv.line（）要好得多，速度更快。
# cv.polylines(img, [pts, pts1, pts2], True, (0, 255, 255), 4)
#
# # 绘制椭圆
# cv.ellipse(img, (256, 256), (100, 50), 0, 0, 270, 255, -1)

# 绘制文字
'''
img：要绘制的图像。
'OpenCV'：要绘制的文本内容。
(10, 500)：文本的起始位置（左下角的坐标）。
font：字体类型。
4：字体大小。
(255, 255, 255)：文本颜色（这里是白色，BGR格式）。
2：文本线条的粗细。
cv.LINE_AA：抗锯齿线条类型，使文本边缘更加平滑。
'''
# font = cv.FONT_HERSHEY_SIMPLEX
# cv.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv.LINE_AA)
#
# cv.imwrite('resource/2.png', img)

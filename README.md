## 基础操作

### 图像基础操作

- 读取图像

> cv.imread('../resource/横须贺2018-6.jpg',0)

> > cv.IMREAD_COLOR：默认参数，以彩色模式加载图像，图像的透明度将被忽略。 1
> >
> > cv.IMREAD_GRAYSCALE：以灰度模式加载图像。 0
> >
> > cv.IMREAD_UNCHANGED：以alpha通道模式加载图像。 -1

- 显示图像

> cv.imshow()

```
 cv.imshow('image',img)
 cv.waitKey(0)
 cv.destroyAllWindows()
```

- 保存图像

```python
cv.imwrite('messigray.png',img)
```

- 使用Matplotlib


OpenCV加载的彩色图像处于**BGR模式**

但Matplotlib以**RGB模式**显示。

因此，如果使用OpenCV读取图像，则Matplotlib中的彩色图像将无法正确显示。

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('messi5.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
```

### 视频基础操作

- 获取视频对象

> cap = cv.VideoCapture(0) 参数为0是默认摄像仪
>
> 参数是"target.mp4"则是视频数据

- 获取视频帧 一帧

> cap.read()

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()    

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv.imshow('frame',gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
```

- 访问帧数据

> cap.get(cv.CAP_PROP_FRAME_WIDTH)
>
> ret=cap.set(cv.CAP_PROP_FRAME_WIDTH，32)

- 保存视频

> 1.首先创建一个VideoWriter对象，我们应该指定输出文件名（例如：output.avi，
> 然后我们应该指定FourCC代码并传递每秒帧数（fps）和帧大小。
> 最后一个是isColor标志，如果是True， 则每一帧是彩色图像，否则每一帧是灰度图像。
>
> 2.确保 VideoWriter 对象的帧率和帧尺寸与捕获的视频匹配，
> 这样可以避免在写入视频文件时出现问题并确保资源在退出时正常释放。

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv.CAP_PROP_FPS)
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
out = cv.VideoWriter('../resource/output.mp4', fourcc, fps, (w, h))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()
cv.waitKey(1) # 确保关闭窗口
```

### 绘制图形

> 可以在图形或者视频中绘制图形

```python
# 绘制直线
cv.line(img,(0,0),(511,511),(255,0,0),5)
# 绘制矩形
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
# 绘制圆形 圆心，半径，BGR 粗度（-1为填充）
cv.circle(img,(447,63), 63, (0,0,255), -1)
# 绘制椭圆
cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
# 绘制多边形 
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2)) # -1 自适应几个点 1 2 每个坐标放在单独的数组里，每个坐标两个值
cv.polylines(img,[pts],True,(0,255,255))
# 绘制文字
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)
```

## 核心操作

### 图像操作

- 常规操作

```python
# 元素点操作
img1 = img.item(10,10,2)
img1 = img.itemset((10,10,2),100)
# 获取图像空间
img1 = img.shape[:2]
# 图像ROI 复制区域
ball = img[280:340, 330:390]
img[273:333, 100:160] = ball
# 处理图像的B、G、R通道
b,g,r = cv.split(img)
img = cv.merge((b,g,r))
# 建议使用索引 
b = img[:,:,0]
# 假设你要将所有红色像素设置为零，则无需先拆分通道。使用Numpy索引更快：
img[:,:,2] = 0
```
- 图像边框
> cv2.BORDER_CONSTANT - 添加一个固定的彩色边框，还需要下一个参数（value）。
>
> cv2.BORDER_REFLECT - 边界元素的镜像。比如: fedcba|abcdefgh|hgfedcb
>
> cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT - 跟上面一样，但稍作改动。例如: gfedcb|abcdefgh|gfedcba
>
> cv2.BORDER_REPLICATE 重复最后一个元素。例如: aaaaaa|abcdefgh|hhhhhhh
>
> cv2.BORDER_WRAP - 不知道怎么说了, 就像这样: cdefgh|abcdefgh|abcdefg

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

BLUE = [255,0,0]

img1 = cv.imread('opencv-logo.png')

replicate = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT)
reflect101 = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT_101)
wrap = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_WRAP)
constant=cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)

plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.show()
```






## 几何变换

### 缩放

> 缩放函数 cv.resize

```python
# 几何变换
# cv.INTER_CUBIC质量更高
# cv.INTER_LINEAR更快，一般使用
# cv.INTER_AREA缩小使用比较好
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('../resource/横须贺2018-6.jpg')

res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)

#OR

height, width = img.shape[:2]
res = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)
# 使用 cv.INTER_AREA 缩小图像
# res = cv.resize(img, None, fx=0.1, fy=0.1, interpolation=cv.INTER_AREA)

plt.subplot(121), plt.imshow(res), plt.title('Input')

```

### 仿射变换

> cv.warpAffine采用2x3变换矩阵作为参数输入

- 平移

> `M = np.float32([[1,0,100],[0,1,50]])`

```python
import numpy as np
import cv2 as cv

img = cv.imread('../resource/横须贺2018-6.jpg',0)
rows,cols = img.shape

M = np.float32([[1,0,100],[0,1,50]])
dst = cv.warpAffine(img,M,(cols,rows))

cv.imshow('img',dst)
cv.waitKey(0)
cv.destroyAllWindows()

```

- 旋转

> M = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)

```python
img = cv.imread('../resource/横须贺2018-6.jpg',0)
rows,cols = img.shape

M = cv.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv.warpAffine(img,M,(cols,rows))
```

- 仿射变换

> `pts1 = np.float32([[50,50],[200,50],[50,200]])`
>
> `pts2 = np.float32([[10,100],[200,50],[100,250]])`
>
> M = cv.getAffineTransform(pts1,pts2)

```python

img = cv.imread('../resource/横须贺2018-6.jpg')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv.getAffineTransform(pts1,pts2)

dst = cv.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
```

### 透视变换

> cv.warpPerspective采用3x3变换矩阵作为参数输入。

```python
img = cv.imread('../resource/横须贺2018-6.jpg')
rows,cols,ch = img.shape

pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

M = cv.getPerspectiveTransform(pts1, pts2)

dst = cv.warpPerspective(res, M, (cols, rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
```


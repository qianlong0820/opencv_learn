import numpy as np
import cv2 as cv

# 创建一个VideoCapture对象 捕获视频对象 0：摄像头数据 '文件名'文件视频数据
# cap = cv.VideoCapture(0)
cap = cv.VideoCapture('resource/target.mp4')
ret = cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
ret = cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)

while (True):
    # 如果读取帧正确，则它将为True。因此，你可以通过值来确定视频的结尾
    ret, frame = cap.read()

    print(cap.isOpened())

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


    # Display the resulting frame
    cv.imshow('frame', gray)
    # print(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    # print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # 视频播放waitKey越小视频越快
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

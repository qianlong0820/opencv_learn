import time

import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv.CAP_PROP_FPS)
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# 检查帧率和帧尺寸是否有效
if fps == 0:
    fps = 20  # 默认帧率


print(f"FPS: {fps}, Width: {w}, Height: {h}")

out = cv.VideoWriter('../resource/output.mp4', fourcc, fps, (w, h))
# out = cv.VideoWriter('../resource/output.mp4', fourcc, 20, (300, 400))


while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # frame = cv.flip(frame, 0)

        # write the flipped frame
        out.write(frame)
        print("Frame written")

        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
print('cap.release()')
out.release()
print('out.release()')
# time.sleep(1000)
# print('time.sleep(1000)')
cv.destroyAllWindows()
print("Frame written ALL")

cv.waitKey(1)
print("Ensure all windows are closed")
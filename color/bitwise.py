import numpy as np
import cv2 as cv

frame1 = np.array([[100, 150, 200],
                    [50, 100, 150],
                    [25, 50, 75]], dtype=np.uint8)

frame2 = np.array([[200, 150, 100],
                    [150, 100, 50],
                    [75, 50, 25]], dtype=np.uint8)

mask = np.array([[0, 255, 0],
                 [255, 0, 255],
                 [0, 255, 0]], dtype=np.uint8)

result = cv.bitwise_and(frame1, frame1, mask=mask)
print(result)

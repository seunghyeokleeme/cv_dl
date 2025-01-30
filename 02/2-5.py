import cv2 as cv
import numpy as np
import sys

cap = cv.VideoCapture(0)

if not cap.isOpened():
    sys.exit('카메라 연결 실패')

frames = []
while True:
    ret, frame = cap.read()

    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    cv.imshow('video display', frame)

    key=cv.waitKey(1)
    if key==ord('c'):
        frames.append(frame)
    elif key==ord('q'):
        break

cap.release()
cv.destroyAllWindows()

if len(frames) > 0:
    imgs = frames[0]
    for i in range(1, min(3, len(frames))):
        imgs=np.hstack((imgs, frames[i]))

    cv.imshow('collected images', imgs)

    cv.waitKey()
    cv.destroyAllWindows()
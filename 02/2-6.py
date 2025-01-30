import cv2 as cv
import sys

img = cv.imread('smile.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

print(img.shape)

cv.rectangle(img, (140, 70), (317, 317), (0, 0, 255), 2)

cv.imshow('Draw', img)

cv.waitKey()
cv.destroyAllWindows()
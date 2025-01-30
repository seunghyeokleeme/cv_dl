import cv2 as cv
import sys

img=cv.imread('test1.png')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

print(type(img))
print(img.shape)

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 컬러 영상을 명암 영상으로 변환하기
gray_small=cv.resize(gray, dsize=(0, 0), fx=0.5, fy=0.5)

cv.imwrite('test_gray.png', gray)
cv.imwrite('test_gray_small.png', gray_small)

cv.imshow('color image', img)
cv.imshow('gray image', gray)
cv.imshow('gray image small', gray_small)

cv.waitKey()
cv.destroyAllWindows()
import cv2 as cv

img=cv.imread('apples.jpg')
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

apples=cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 100, param1=150, param2=10, minRadius=20, maxRadius=100)

for i in apples[0]:
    cv.circle(img, (int(i[0]),int(i[1])), int(i[2]), (255, 0, 0), 2)

cv.imshow('Apple detection', img)

cv.waitKey()
cv.destroyAllWindows()
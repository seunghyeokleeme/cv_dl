import cv2 as cv
import sys

img = cv.imread('smile.jpg')

if img is None:
    sys.exit('파일을 찾을수 없습니다.')

def draw(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.rectangle(img, (x, y), (x+100, y+100), (0, 0, 255), 2)
    elif event == cv.EVENT_RBUTTONDOWN:
        cv.rectangle(img, (x,y), (x+100, y+100), (255, 0, 0), 2)
    
    cv.imshow('Drawing', img)

cv.namedWindow('Drawing')
cv.imshow('Drawing', img)

cv.setMouseCallback('Drawing', draw)

while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break
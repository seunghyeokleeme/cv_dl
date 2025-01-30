import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('test1.png')
if img is None:
	print("Image file 'test1.png' not found or cannot be read.")
h = cv.calcHist([img], [2], None, [256], [0, 256])

plt.plot(h, color='r')
plt.show()
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('img/j.png')
kernel = np.ones((5,5), np.uint8)
erosion = cv.erode(img, kernel, iterations = 1)
dilation = cv.dilate(img, kernel, iterations = 1)
while 0xFF & cv.waitKey(1) != ord('q'):
    cv.imshow('Original', img)
    cv.imshow('Erosion', erosion)
    cv.imshow('Dilation', dilation)
cv.destroyAllWindows()

img2 = cv.imread('img/j_salt.png')
opening = cv.morphologyEx(img2, cv.MORPH_OPEN, kernel)
while 0xFF & cv.waitKey(1) != ord('q'):
    cv.imshow('Salt', img2)
    cv.imshow('Opening', opening)
cv.destroyAllWindows()

img3 = cv.imread('img/j_pepper.png')
closing = cv.morphologyEx(img3, cv.MORPH_CLOSE, kernel)
while 0xFF & cv.waitKey(1) != ord('q'):
    cv.imshow('Pepper', img3)
    cv.imshow('Closing',closing)
cv.destroyAllWindows()

kernel2 = np.ones((9,9), np.uint8)
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel2)
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel2)
while 0xFF & cv.waitKey(1) != ord('q'):
    cv.imshow('Original', img)
    cv.imshow('Gradient', gradient)
    cv.imshow('Top Hat', tophat)
    cv.imshow('Black Hat', blackhat)
cv.destroyAllWindows()

rectangle = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
cross = cv.getStructuringElement(cv.MORPH_CROSS, (5,5))
plt.subplot(131)
plt.imshow(rectangle, cmap = 'gray')
plt.subplot(132)
plt.imshow(ellipse, cmap = 'gray')
plt.subplot(133)
plt.imshow(cross, cmap = 'gray')
plt.show()

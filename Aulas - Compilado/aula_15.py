import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def opening_reconstruction(seeds, mask):
    previous = np.zeros_like(seeds)
    current = np.copy(seeds)
    while not np.array_equal(previous, current):
        previous = current
        current = cv.dilate(current, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7)), iterations=5)
        current = cv.bitwise_and(current, mask)
    return current

img = cv.imread('img/calculator.tif', cv.IMREAD_GRAYSCALE)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
current = np.copy(img)
for i in range(10):
    erosion = cv.erode(current, kernel, iterations = 1)
    current = np.copy(erosion)
im2 = opening_reconstruction(current, img)
#tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
plt.subplot('221')
plt.imshow(img, cmap = 'gray')
plt.subplot('222')
plt.imshow(im2, cmap = 'gray')
#plt.imshow(opening, cmap = 'gray')
#plt.subplot('223')
#plt.imshow(tophat, cmap = 'gray')
#plt.subplot('224')
#plt.imshow(img - (img - opening), cmap = 'gray')
plt.show()

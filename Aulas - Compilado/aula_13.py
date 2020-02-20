import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

noisy = cv.imread('img/noisy-fingerprint.tif', cv.IMREAD_GRAYSCALE)
kernel = np.ones((3,3), np.uint8)
noisy_o = cv.morphologyEx(noisy, cv.MORPH_OPEN, kernel)
noisy_oc = cv.morphologyEx(noisy_o, cv.MORPH_CLOSE, kernel)

plt.subplot(131)
plt.imshow(noisy, cmap = 'gray')
plt.title('Original')
plt.subplot(132)
plt.imshow(noisy_o, cmap = 'gray')
plt.title('Opened')
plt.subplot(133)
plt.imshow(noisy_oc, cmap = 'gray')
plt.title('Opened and closed')
plt.show()

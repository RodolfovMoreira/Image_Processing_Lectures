import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

input_img = np.array(([0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 255, 255, 255, 0, 0, 0, 255],
                        [0, 255, 255, 255, 0, 0, 0, 0],
                        [0, 255, 255, 255, 0, 255, 0, 0],
                        [0, 0, 255, 0, 0, 0, 0, 0],
                        [0, 0, 255, 0, 0, 255, 255, 0],
                        [0, 255, 0, 255, 0, 0, 255, 0],
                        [0, 255, 255, 255, 0, 0, 0, 0]), dtype = "uint8")

kernel = np.array(([0, 1, 0], [-1, 1, 1], [-1, -1, 0]), dtype = "int")

output_img = cv.morphologyEx(input_img, cv.MORPH_HITMISS, kernel)

plt.subplot(131)
plt.imshow(input_img, cmap = 'gray')
plt.title('Input')
plt.subplot(132)
plt.imshow(kernel, cmap = 'gray')
plt.title('Kernel')
plt.subplot(133)
plt.imshow(output_img, cmap = 'gray')
plt.title('Output')
plt.show()

#%%
a = cv.imread('img/lincoln.tif', cv.IMREAD_GRAYSCALE)
b = np.ones((3,3), np.uint8)
c = cv.morphologyEx(a, cv.MORPH_DILATE, b)
d = c & ~a

plt.subplot(131)
plt.imshow(a, cmap = 'gray')
plt.title('Input')
plt.subplot(132)
plt.imshow(c, cmap = 'gray')
plt.title('Dilatacao')
plt.subplot(133)
plt.imshow(d, cmap = 'gray')
plt.title('Output')
plt.show()
#%%
img = cv.imread('img/region-filling-reflections.tif', cv.IMREAD_GRAYSCALE)
mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)

col = [0, 160, 180, 300]
row = [0, 250, 200, 240]
result = img.copy()
idx = 0
cv.floodFill(result, mask, (row[idx], col[idx]), 255)
result_inv = cv.bitwise_not(result)
output = img | result_inv

plt.subplot(231)
plt.imshow(img, cmap = 'gray')
plt.title('Input')
plt.subplot(232)
plt.imshow(~img, cmap = 'gray')
plt.title('Inverse')
plt.subplot(233)
plt.imshow(result, cmap = 'gray')
plt.title('Result')
plt.subplot(234)
plt.imshow(result_inv, cmap = 'gray')
plt.title('Inverse Result')
plt.subplot(235)
plt.imshow(output, cmap = 'gray')
plt.title('Output')
plt.show()

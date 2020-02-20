# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:50:23 2019

@author: horacio
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

noisy = cv2.imread(os.path.join('/home/horacio/Imagens/PDI', 'noisy-fingerprint.tif'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3,3), np.uint8)
noisy_o = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel)
noisy_oc = cv2.morphologyEx(noisy_o, cv2.MORPH_CLOSE, kernel)

plt.subplot(131), plt.imshow(noisy, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(noisy_o, cmap='gray'), plt.title('Noisy_o')
plt.subplot(133), plt.imshow(noisy_oc, cmap='gray'), plt.title('Noisy_oc')
plt.show();

#%%

import cv2
import numpy as np
import matplotlib.pyplot as plt

input_image = np.array((
    [0,0,0,0,0,0,0,0],
    [0,255,255,255,0,0,0,255],
    [0,255,255,255,0,0,0,0],
    [0,255,255,255,0,255,0,0],
    [0,0,255,0,0,0,0,0],
    [0,0,255,0,0,255,255,0],
    [0,255,0,255,0,0,255,0],
    [0,255,255,255,0,0,0,0]), dtype="uint8")

k = np.array((
    [0,1,0],
    [-1,1,1],
    [-1,-1,0]), dtype="int")

output_image = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, k)

plt.subplot(131), plt.imshow(input_image, cmap='gray'), plt.title('I')
plt.subplot(132), plt.imshow(k, cmap = 'gray'), plt.title('k')
plt.subplot(133), plt.imshow(output_image, cmap = 'gray'), plt.title('')
plt.show()

#%% BOUNDARY EXTRACTION
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

a = cv2.imread(os.path.join('/home/horacio/Imagens/PDI', 'lincoln.tif'), cv2.IMREAD_GRAYSCALE)
b = np.ones((3,3), np.uint8)
c = cv2.morphologyEx(a, cv2.MORPH_DILATE, b)
d = c & ~a

plt.subplot(131), plt.imshow(a, cmap='gray'), plt.title('A')
plt.subplot(132), plt.imshow(c, cmap = 'gray'), plt.title('$C = A \ominus B$')
plt.subplot(133), plt.imshow(d, cmap = 'gray'), plt.title('$D = A - (A \ominus B)$')
plt.show()

# %% HOLE FILLING
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(os.path.join('/home/horacio/Imagens/PDI', 'region-filling-reflections.tif'), cv2.IMREAD_GRAYSCALE)
mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)

col = [0, 160, 180, 300]
row = [0, 250, 200, 240]
result = img.copy()
idx = 1
cv2.floodFill(result, mask, (row[idx], col[idx]), 255)
result_inv = cv2.bitwise_not(result)
output = img | result_inv

plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('A')
plt.subplot(232), plt.imshow(~img, cmap = 'gray'), plt.title('$A^C$')
plt.subplot(233), plt.imshow(result, cmap = 'gray'), plt.title('$R = (X_{k-1} \oplus B)$')
plt.subplot(234), plt.imshow(result_inv, cmap='gray'), plt.title('$R^C$')
plt.subplot(235), plt.imshow(output, cmap = 'gray'), plt.title('Mask')
plt.show()



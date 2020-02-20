# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 09:31:36 2019

@author: horacio
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/home/horacio/Imagens/PDI/chips.png', cv2.IMREAD_COLOR)

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
hsv = cv2.split(img2)

while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow('img', img)
    cv2.imshow('img2', img2)
    cv2.imshow('h', hsv[0])
    cv2.imshow('s', hsv[1])
    cv2.imshow('v', scaleImage2_uchar(hsv[2]))
cv2.destroyAllWindows()

#imgplot = plt.imshow(img, clim=(0.0, 0.7))

#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('/home/horacio/Imagens/PDI/chips.png', cv2.IMREAD_COLOR)

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv = cv2.split(img2)

#plt.subplot('220'), plt.imshow( brg2rgb(img) )
plt.subplot('221'), plt.imshow( (img) )
plt.subplot('222'), plt.imshow(hsv[0], cmap='gray')
plt.subplot('223'), plt.imshow(hsv[1], cmap='gray')
plt.subplot('224'), plt.imshow(hsv[2], cmap='gray')
plt.show()

#%%


#%% BORRAMENTO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função doNothing
def doNothing():
    return

img = cv2.imread('/home/horacio/Imagens/PDI/baboon.png', cv2.IMREAD_COLOR)

xsize = 3
ysize = 3
cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('img2', cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('xsize', 'img2', xsize, 50, doNothing)
cv2.createTrackbar('ysize', 'img2', ysize, 50, doNothing)

while cv2.waitKey(1) != ord('q'):
    xsize = cv2.getTrackbarPos('xsize', 'img2')
    ysize = cv2.getTrackbarPos('ysize', 'img2')
    
    img2 = cv2.blur(img, (xsize + 1, ysize + 1))
    cv2.imshow('img', img)
    cv2.imshow('img2', img2)
cv2.destroyAllWindows()

#%% AGUÇAMENTO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função doNothing
def doNothing():
    return

img = cv2.imread('/home/horacio/Imagens/PDI/baboon.png', cv2.IMREAD_COLOR)

wsize = 3
cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('img2', cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('xsize', 'img2', wsize, 10, doNothing)

while cv2.waitKey(1) != ord('q'):
    wsize = cv2.getTrackbarPos('wsize', 'img2')
    
    img2 = cv2.Laplacian(img, cv2.CV_16S, 
                         ksize=2*wsize+1,
                         scale=1, 
                         delta=0, 
                         borderType=cv2.BORDER_DEFAULT)
    cv2.imshow('img', img)
    cv2.imshow('img2', scaleImage2_uchar(img2))
cv2.destroyAllWindows()

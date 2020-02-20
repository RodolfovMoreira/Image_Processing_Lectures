import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('chips.png', cv2.IMREAD_COLOR)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv = cv2.split(img2)
plt.subplot('221')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Imagem")
plt.subplot('222')
plt.imshow(hsv[0], cmap='gray')
plt.title("Matiz")
plt.subplot('223')
plt.imshow(hsv[1], cmap='gray')
plt.title("Saturacao")
plt.subplot('224')
plt.imshow(hsv[2], cmap='gray')
plt.title("Valor")
plt.show()

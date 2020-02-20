#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
#%%
img = cv2.imread("raio.jpg", cv2.IMREAD_GRAYSCALE)
#cv2.imshow("raio", img)
#cv2.waitKey(0)
#cv2.destroyAllWIndows()
#img = img + 100
plt.subplot('211')
plt.title('Original')
plt.imshow(img, 'gray')
plt.subplot('212')
plt.title('Histogram')
plt.hist(img.ravel(), 256, [0, 256])
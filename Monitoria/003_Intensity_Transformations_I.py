import cv2
import matplotlib.pyplot as plt
import numpy as np

def doNothing(x):
    pass

#%%


# Intensity Transformations - Part I


# Image negative
 img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
 cv2.imshow("img", 255 - img)
 cv2.waitKey(0)
 cv2.destroyAllWindows()

#%%

# Log transform
 img = cv2.imread("img/spectrum.tif", cv2.IMREAD_GRAYSCALE)
 img2 = np.ones(img.shape, np.float64)
 c = 1
 for x in range(img.shape[0]):
     for y in range(img.shape[1]):
         intensity = img[x][y]
         intensity_new = c * np.log(1 + intensity)
         img2[x][y] = intensity_new

 cv2.normalize(img2, img2, 1, 0, cv2.NORM_MINMAX)

 cv2.imshow("img", img)
 cv2.imshow("img2", img2)

 cv2.waitKey(0)
 cv2.destroyAllWindows()

#%%

# Intensity transform
 img = cv2.imread("img/spectrum.tif", cv2.IMREAD_GRAYSCALE)
 img2 = np.ones(img.shape, np.uint8)

 cv2.namedWindow("img2", cv2.WINDOW_KEEPRATIO)

 n = 0
 cv2.createTrackbar("n", "img2", n, 4, doNothing)

 while cv2.waitKey(1) != ord('q'):

     n = cv2.getTrackbarPos("n", "img2")

     for x in range(img.shape[0]):
         for y in range(img.shape[1]):
             intensity = img[x][y]
             intensity_new = np.power(intensity, n)
             img2[x][y] = intensity_new

     cv2.imshow("img", img)
     cv2.imshow("img2", img2)
 cv2.destroyAllWindows()



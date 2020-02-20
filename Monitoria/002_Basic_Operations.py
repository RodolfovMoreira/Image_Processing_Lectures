import cv2
import matplotlib.pyplot as plt
import numpy as np

#%%

#
# LINEAR AND NONLINEAR OPERATIONS
#

# Add a scalar to an image
 img = cv2.imread("lena.png", cv2.IMREAD_COLOR)
 val = 100
 img2 = img + val
 plt.subplot("121"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 plt.subplot("122"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
 plt.show()


#%%
# Add two images
 img = cv2.imread("lena.png", cv2.IMREAD_COLOR)
 img2 = cv2.imread("baboon.png", cv2.IMREAD_COLOR)
 img3 = img + img2
 plt.subplot("131"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 plt.subplot("132"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
 plt.subplot("133"); plt.title("IMG 3"); plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
 plt.show()

#%%

# Max operator
 img = cv2.imread("lena.png", cv2.IMREAD_COLOR)
 img2 = cv2.imread("baboon.png", cv2.IMREAD_COLOR)
 img3 = cv2.max(img, img2)
 plt.subplot("131"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 plt.subplot("132"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
 plt.subplot("133"); plt.title("IMG 3"); plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
 plt.show()

#%%

# Absolute image diferencing
 img = cv2.imread("lena.png", cv2.IMREAD_COLOR)
 img2 = cv2.imread("baboon.png", cv2.IMREAD_COLOR)
 img3 = cv2.absdiff(img, img2)
 plt.subplot("131"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 plt.subplot("132"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
 plt.subplot("133"); plt.title("IMG 3"); plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
 plt.show()

#%%

# Image difference
 cap = cv2.VideoCapture(0)

 while cv2.waitKey(1) != ord('q'):
     _, frame1 = cap.read()
     _, frame2 = cap.read()

     gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
     gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
     diff = cv2.absdiff(gray1, gray2)
     cv2.imshow('Gray 1', gray1)
     cv2.imshow('Gray 2', gray2)
     cv2.imshow('DIFF', diff)

 cap.release()
 cv2.destroyAllWindows()

#%%

# Adding noise to an image
 img = cv2.imread("lena.png", cv2.IMREAD_COLOR)
 noise = np.zeros(img.shape, img.dtype)
 cv2.randn(noise, 0, 150)
 img2 = img + noise
 plt.subplot("121"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 plt.subplot("122"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
 plt.show()

#%%

# Adding salt & pepper noise to an image
 img = cv2.imread("lena.png", cv2.IMREAD_COLOR)
 noise = np.zeros((img.shape[0], img.shape[1]), img.dtype)
 cv2.randu(noise, 0, 255)
 salt = noise > 250
 pepper = noise < 5
 img2 = img.copy()
 img2[salt == True] = 255
 img2[pepper == True] = 0
 plt.subplot("121"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 plt.subplot("122"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
 plt.show()

#%%

# Set operations
 img = cv2.imread("img/utk.tif", cv2.IMREAD_GRAYSCALE)
 img2 = cv2.imread("img/gt.tif", cv2.IMREAD_GRAYSCALE)
 and_img = img & img2
 or_img = img | img2
 not_img = ~img
 plt.subplot("151"); plt.title("IMG 1"); plt.imshow(img, 'gray')
 plt.subplot("152"); plt.title("IMG 2"); plt.imshow(img2, 'gray')
 plt.subplot("153"); plt.title("AND"); plt.imshow(and_img, 'gray')
 plt.subplot("154"); plt.title("OR"); plt.imshow(or_img, 'gray')
 plt.subplot("155"); plt.title("NOT"); plt.imshow(not_img, 'gray')
 plt.show()
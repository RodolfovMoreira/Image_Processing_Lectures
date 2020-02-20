import cv2
import numpy as np

img = cv2.imread('rectangle.jpg', cv2.IMREAD_GRAYSCALE)
planes = [np.zeros(img.shape, dtype = np.float64), np.zeros(img.shape, dtype = np.float64)]
planes[0][:] = np.float64(img[:])
planes[1][:] = np.float64(img[:])
cv2.normalize(planes[0], planes[0], 1, 0, cv2.NORM_MINMAX)
cv2.normalize(planes[1], planes[1], 1, 0, cv2.NORM_MINMAX)
img2 = cv2.merge(planes)
img2 = cv2.dft(img2)
planes = cv2.split(img2)
magnitude_spectrum = cv2.magnitude(planes[0], planes[1])
magnitude_spectrum += 1
magnitude_spectrum = np.log(magnitude_spectrum)
cv2.normalize(magnitude_spectrum, magnitude_spectrum, 1, 0, cv2.NORM_MINMAX)
while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow('Original', img)
    cv2.imshow('FFT Shift', magnitude_spectrum)
cv2.destroyAllWindows()

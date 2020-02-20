import cv2
import numpy as np

img = np.array(cv2.imread("baboon.png"))
cut_img = img[256:478, 128:350]
cv2.imshow("Antes", img)
cv2.imshow("Resultado", cut_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
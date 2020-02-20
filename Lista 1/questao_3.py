import cv2
import numpy as np

img = np.array(cv2.imread("lena.png"))
lighter_img = np.where((255 - img) < 64, 255, img + 64)
darker_img = np.where(img < 64, 0, img - 64)
cv2.imshow("Mais escura", darker_img)
cv2.imshow("Normal", img)
cv2.imshow("Mais clara", lighter_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
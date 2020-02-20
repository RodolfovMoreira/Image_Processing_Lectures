import cv2
import numpy as np

black = np.zeros((400, 400,3))
grey = np.ones((400, 400, 3)) * 128 / 255.
cv2.imshow("Black", black)
cv2.imshow("Grey", grey)
cv2.waitKey(0)
cv2.destroyAllWindows()
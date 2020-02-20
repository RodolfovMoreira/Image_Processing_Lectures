#%%
import numpy as np
import cv2 

#%%
#cv2._version_
img = cv2.imread("raio.jpg", cv2.IMREAD_COLOR)
cv2.imshow("Raio", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
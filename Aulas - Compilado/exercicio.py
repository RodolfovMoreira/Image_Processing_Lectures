import cv2 as cv
import numpy as np
#%%
flags = [i for i in dir(cv) if i.startswith('COLOR_')]
print(flags)
#%%
cap = cv.VideoCapture(0)

while(1):
    _, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_blue = np.array([0, 50, 50])
    upper_blue = np.array([10, 255, 255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    res = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv.destroyAllWindows()
#%%
green = np.uint8([[[0, 0, 255]]])
hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
print hsv_green
from dip import *

img = cv2.imread('baboon.png', cv2.IMREAD_COLOR)
xsize = 3
ysize = 3
cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('img2', cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('xsize','img2', xsize, 50, doNothing)
cv2.createTrackbar('ysize','img2', ysize, 50, doNothing)

while cv2.waitKey(1) != ord('q'):
    xsize = cv2.getTrackbarPos('xsize', 'img2')
    ysize = cv2.getTrackbarPos('ysize', 'img2')
    img2 = cv2.blur(img, (xsize+1, ysize+1))
    cv2.imshow('img', img)
    cv2.imshow('img2', img2)
cv2.destroyAllWindows()

img = cv2.imread('baboon.png', cv2.IMREAD_COLOR)
wsize = 3
cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('img2', cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar('wsize','img2', wsize, 10, doNothing)
while cv2.waitKey(1) != ord('q'):
    wsize = cv2.getTrackbarPos('wsize', 'img2')
    img2 = cv2.Laplacian(img, cv2.CV_16S, ksize = 2*wsize+1, scale = 1, delta = 0, borderType = cv2.BORDERDEFAULT)
    cv2.imshow('img', img)
    cv2.imshow('img2', scaleImage2_uchar(img2))
cv2.destroyAllWindows()

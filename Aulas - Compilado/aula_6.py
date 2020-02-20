from dip import *
#%%
rows = 100
cols = 100
theta = 0
xc = 50
yc = 50
sx = 30
sy = 10

cv.namedWindow('img', cv.WINDOW_KEEPRATIO)
cv.createTrackbar('xc', 'img', xc, int(rows), doNothing)
cv.createTrackbar('yc', 'img', yc, int(rows), doNothing)
cv.createTrackbar('sx', 'img', sx, int(rows), doNothing)
cv.createTrackbar('sy', 'img', sy, int(rows), doNothing)
cv.createTrackbar('theta', 'img', theta, 360, doNothing)
while 0xFF & cv.waitKey(1) != ord('q'):
    xc = cv.getTrackbarPos('xc', 'img')
    yc = cv.getTrackbarPos('yc', 'img')
    sx = cv.getTrackbarPos('sx', 'img')
    sy = cv.getTrackbarPos('sy', 'img')
    theta = cv.getTrackbarPos('theta', 'img')
    img = create2DGaussian(rows, cols, xc, yc, sx, sy, theta)
    cv.imshow('img', cv.applyColorMap(scaleImage2_uchar(img), cv.COLORMAP_JET))
cv.destroyAllWindows()
#%%
img = cv.imread('rectangle.jpg', cv.IMREAD_GRAYSCALE)
cv.namedWindow('Original', cv.WINDOW_KEEPRATIO)
cv.namedWindow('Plane 0 - Real', cv.WINDOW_KEEPRATIO)
cv.namedWindow('Plane 1 - Imaginary', cv.WINDOW_KEEPRATIO)

planes = [np.zeros(img.shape, dtype = np.float64), np.zeros(img.shape, dtype = np.float64)]
planes[0][:] = np.float64(img[:])
img2 = cv.merge(planes)
img2 = cv.dft(img2)
planes = cv.split(img2)
cv.normalize(planes[0], planes[0], 1, 0, cv.NORM_MINMAX)
cv.normalize(planes[1], planes[1], 1, 0, cv.NORM_MINMAX)
while 0xFF & cv.waitKey(1) != ord('q'):
    cv.imshow('Original', img)
    cv.imshow('Plane 0 - Real', planes[0])
    cv.imshow('Plane 1 - Imaginary', planes[1])
cv.destroyAllWindows()

import cv2
import numpy as np


def doNothing(x):
    pass


def scaleImage2_uchar(src):
    tmp = np.copy(src)
    if src.dtype != np.float32:
        tmp = np.float32(tmp)
    cv2.normalize(tmp, tmp, 1, 0, cv2.NORM_MINMAX)
    tmp = 255 * tmp
    tmp = np.uint8(tmp)
    return tmp


def compute_histogram_1C(src):
    # Compute the histograms:
    b_hist = cv2.calcHist([src], [0], None, [256], [0, 256], True, False)

    # Draw the histograms for B, G and R
    hist_w = 512
    hist_h = 400
    bin_w = np.round(hist_w / 256)

    histImage = np.ones((hist_h, hist_w), np.uint8)

    # Normalize the result to [ 0, histImage.rows ]
    cv2.normalize(b_hist, b_hist, 0, histImage.shape[0], cv2.NORM_MINMAX)

    # Draw for each channel
    for i in range(1, 256):
        cv2.line(histImage, (int(bin_w * (i - 1)), int(hist_h - np.round(b_hist[i - 1]))),
                 (int(bin_w * i), int(hist_h - np.round(b_hist[i]))), 255, 2, cv2.LINE_8, 0)

    return histImage

#%%
# Average Blurring

 img = cv2.imread('lena.png')

 cv2.namedWindow("Original", cv2.WINDOW_KEEPRATIO)
 cv2.namedWindow("New", cv2.WINDOW_KEEPRATIO)

 ksizex = 0;
 ksizey = 0

 cv2.createTrackbar("ksizex", "New", ksizex, 63, doNothing)
 cv2.createTrackbar("ksizey", "New", ksizey, 63, doNothing)

 img2 = np.zeros(img.shape, dtype=np.float64)

 while cv2.waitKey(1) != ord('q'):

     ksizey = cv2.getTrackbarPos("ksizey", "New")
     ksizex = cv2.getTrackbarPos("ksizex", "New")

     if ksizex < 1:
         ksizex = 1
     if ksizey < 1:
         ksizey = 1

     img2 = cv2.blur(img, (ksizex, ksizey), img2, (-1, -1), cv2.BORDER_DEFAULT)

     cv2.imshow("Original", img)
     cv2.imshow("New", img2)

cv2.destroyAllWindows()


#%%

# Adding salt & pepper noise to an image and cleaning it using the median
 img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

 noise = np.zeros(img.shape, np.uint8)
 img2 = np.zeros(img.shape, np.uint8)
 img3 = np.zeros(img.shape, np.uint8)
 salt = np.zeros(img.shape, np.uint8)
 pepper = np.zeros(img.shape, np.uint8)

 ksize = 0
 amount = 5
 cv2.namedWindow("img3", cv2.WINDOW_KEEPRATIO);
 cv2.namedWindow("img2", cv2.WINDOW_KEEPRATIO);
 cv2.createTrackbar("ksize", "img3", ksize, 15, doNothing)
 cv2.createTrackbar("amount", "img2", amount, 120, doNothing)

 cv2.randu(noise, 0, 255)

 while cv2.waitKey(1) != ord('q'):
     amount = cv2.getTrackbarPos("amount", "img2")
     ksize = cv2.getTrackbarPos("ksize", "img3")

     img2 = np.copy(img)

     salt = noise > 255 - amount
     pepper = noise < amount

     img2[salt == True] = 255
     img2[pepper == True] = 0

     img3 = cv2.medianBlur(img2, (ksize + 1) * 2 - 1)

     cv2.imshow("img", img)
     cv2.imshow("img2", img2)
     cv2.imshow("img3", img3)


cv2.destroyAllWindows()


#%%

# First derivative operators - Sobel masks - Part I
# The Sobel operator performs a 2-D spatial gradient measurement on an image and so emphasizes regions of 
# high spatial frequency that correspond to edges. 
# Typically it is used to find the approximate absolute gradient magnitude at each point in an input grayscale image.

 cv2.namedWindow("Original", cv2.WINDOW_KEEPRATIO)
 cv2.namedWindow("New", cv2.WINDOW_KEEPRATIO)
 cv2.namedWindow("Gx", cv2.WINDOW_KEEPRATIO)
 cv2.namedWindow("Gy", cv2.WINDOW_KEEPRATIO)

 img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

 img2 = np.float64(img)

 for x in range(img.shape[0]):
     for y in range(img.shape[1]):
         img2[x,y] = np.float64(img[x,y])

 kx = [[-1,  0,  1],
       [-2,  0,  2],
       [-1,  0,  1]]
 kx = np.array(kx)

 ky = [[-1, -2, -1],
       [ 0,  0,  0],
       [ 1,  2,  1]]
 ky = np.array(ky)

 gx = cv2.filter2D(img2, -1, kx, cv2.BORDER_DEFAULT)
 gy = cv2.filter2D(img2, -1, ky, cv2.BORDER_DEFAULT)
 g = np.abs(gx) + np.abs(gy)

 cv2.normalize(gx, gx, 1, 0, cv2.NORM_MINMAX)
 cv2.normalize(gy, gy, 1, 0, cv2.NORM_MINMAX)
 cv2.normalize(g, g, 1, 0, cv2.NORM_MINMAX)


 while cv2.waitKey(1) != ord('q'):
     cv2.imshow("Original", img)
     cv2.imshow("New", g)
     cv2.imshow("Gx", gx)
     cv2.imshow("Gy", gy)

cv2.destroyAllWindows()


#%%

# First derivative operators - Sobel masks - Part II

 img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

 gx, gy = cv2.spatialGradient(img, ksize=3, borderType=cv2.BORDER_DEFAULT)
 g = np.abs(gx) + np.abs(gy)

 gx = scaleImage2_uchar(gx)
 gy = scaleImage2_uchar(gy)
 g = scaleImage2_uchar(g)

 while cv2.waitKey(1) != ord('q'):
     cv2.imshow("Original", img)
     cv2.imshow("New", g)
     cv2.imshow("Gx", gx)
     cv2.imshow("Gy", gy)


cv2.destroyAllWindows()


#%%

# First derivative operators - Sobel masks - Part III

 img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

 gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
 gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
 g = np.abs(gx) + np.abs(gy)

 gx = scaleImage2_uchar(gx)
 gy = scaleImage2_uchar(gy)
 g = scaleImage2_uchar(g)

 while cv2.waitKey(1) != ord('q'):
     cv2.imshow("Original", img)
     cv2.imshow("New", g)
     cv2.imshow("Gx", gx)
     cv2.imshow("Gy", gy)

cv2.destroyAllWindows()


#%%

# Image sharpening using the Laplacian operator - Part I

# The Laplacian of an image highlights regions of rapid intensity change and is therefore often used for edge detection 
# The Laplacian is often applied to an image that has first been smoothed with something approximating a Gaussian smoothing 
# filter in order to reduce its sensitivity to noise, and hence the two variants will be described together here. 
# The operator normally takes a single graylevel image as input and produces another graylevel image as output.

 cv2.namedWindow("img3", cv2.WINDOW_KEEPRATIO)

 img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

 img = np.float32(img)

 kernel = [[1.0,  1.0, 1.0],
           [1.0, -8.0, 1.0],
           [1.0,  1.0, 1.0]]
 kernel = np.array(kernel)

 img2 = cv2.filter2D(img, -1, kernel, cv2.BORDER_DEFAULT)
 img3 = np.copy(img2)

 cv2.normalize(img2, img2, 1, 0, cv2.NORM_MINMAX)

 factor = 5
 cv2.createTrackbar("factor", "img3", factor, 500, doNothing)

 while cv2.waitKey(1) != ord('q'):
     factor = cv2.getTrackbarPos("factor", "img3")

     img3 = img - factor * img2
     hist = compute_histogram_1C(img3)

     cv2.imshow("img", scaleImage2_uchar(img))
     cv2.imshow("img2", scaleImage2_uchar(img2))
     cv2.imshow("img3", scaleImage2_uchar(img3))
     cv2.imshow("hist", hist)

cv2.destroyAllWindows()


#%%

# Image sharpening using the Laplacian operator - Part II
 img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
 lap = cv2.Laplacian(img, ddepth=cv2.CV_32F, ksize=1, scale=1, delta=0)

 img = np.float32(img)
 lap = np.float32(lap)

 cv2.normalize(img, img, 1, 0, cv2.NORM_MINMAX)

 img2 = img - lap

 while cv2.waitKey(1) != ord('q'):
     cv2.imshow("img", img)
     cv2.imshow("img2", img2)
     cv2.imshow("lap", scaleImage2_uchar(lap))

cv2.destroyAllWindows()


#%%

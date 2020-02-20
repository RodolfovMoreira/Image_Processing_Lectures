import cv2
import matplotlib.pyplot as plt
import numpy as np


def doNothing(x):
    pass


def compute_piecewise_linear_val(val, r1, s1, r2, s2):
    output = 0
    if (0 <= val) and (val <= r1):
        output = (s1 / r1) * val
    if (r1 <= val) and (val <= r2):
        output = ((s2 - s1) / (r2 - r1)) * (val - r1) + s1
    if (r2 <= val) and (val <= 1):
        output = ((1 - s2) / (1 - r2)) * (val - r2) + s2

    return output


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

# Pice-wise linear transformation
 #img = cv2.imread("img/grains.jpg", cv2.IMREAD_GRAYSCALE)
 img = cv2.imread("img/kidney.tif", cv2.IMREAD_GRAYSCALE)
 img = cv2.imread("img/aerial.tif", cv2.IMREAD_GRAYSCALE)
 img = cv2.imread("img/spine.jpg", cv2.IMREAD_GRAYSCALE)

 img = cv2.imread("img/pollen_washedout.tif", cv2.IMREAD_GRAYSCALE)
 img = cv2.resize(img, (256, 256), 0, 0, cv2.INTER_LINEAR)
 img2 = np.copy(img)
 hist = np.copy(img)
 hist2 = np.copy(img)
 T0 = 255 * np.ones(img.shape, np.uint8)

 cv2.namedWindow("Transformation", cv2.WINDOW_AUTOSIZE)

 r1 = 65
 s1 = 65
 r2 = 195
 s2 = 195

 cv2.createTrackbar("r1", "Transformation", r1, T0.shape[0] - 1, doNothing)
 cv2.createTrackbar("s1", "Transformation", s1, T0.shape[0] - 1, doNothing)
 cv2.createTrackbar("r2", "Transformation", r2, T0.shape[0] - 1, doNothing)
 cv2.createTrackbar("s2", "Transformation", s2, T0.shape[0] - 1, doNothing)

 while True:

     r1 = cv2.getTrackbarPos("r1", "Transformation")
     s1 = cv2.getTrackbarPos("s1", "Transformation")
     r2 = cv2.getTrackbarPos("r2", "Transformation")
     s2 = cv2.getTrackbarPos("s2", "Transformation")

     T = np.copy(T0)
     p1 = (r1, T.shape[1] - 1 - s1)
     p2 = (r2, T.shape[1] - 1 - s2)
     cv2.line(T, (0, T.shape[0] - 1), p1, (0, 0, 0), 2, cv2.LINE_8, 0)
     cv2.circle(T, p1, 4, 0, 2, cv2.LINE_8, 0)
     cv2.line(T, p1, p2, 0, 2, cv2.LINE_8, 0)
     cv2.circle(T, p2, 4, 0, 2, cv2.LINE_8, 0)
     cv2.line(T, p2, (T.shape[0] - 1, 0), 0, 2, cv2.LINE_8, 0)

     r = 0
     s = 0

     for x in range(img.shape[0]):
         for y in range(img.shape[1]):
             r = img[y][x]
             s = compute_piecewise_linear_val(r / 255.0, r1 / 255.0, s1 / 255.0, r2 / 255.0, s2 / 255.0)
             img2[y][x] = 255.0 * s

     hist = compute_histogram_1C(img)
     hist2 = compute_histogram_1C(img2)
     cv2.imshow("img", img)
     cv2.imshow("img2", img2)
     cv2.imshow("hist", hist)
     cv2.imshow("hist2", hist2)
     cv2.imshow("Transformation", T)
     if cv2.waitKey(1) == ord('q'):
         break
     
cv2.destroyAllWindows()


#%%
# Thresholding
 img = cv2.imread("img/kidney.tif", cv2.IMREAD_GRAYSCALE)
 for x in range(img.shape[0]):
     for y in range(img.shape[1]):
         img[x][y] = img[x][y] if (150 < img[x][y] < 240) else 0
 cv2.imshow("img", img)
 cv2.waitKey(0)
   
cv2.destroyAllWindows()

#%%

# Thresholding
 cv2.namedWindow("result", cv2.WINDOW_KEEPRATIO)
 lower = 0
 upper = 255
 cv2.createTrackbar("lower", "result", lower, 255, doNothing)
 cv2.createTrackbar("upper", "result", upper, 255, doNothing)

 # img = cv2.imread("img/breast.tif", cv2.IMREAD_GRAYSCALE)
 img = cv2.imread("img/kidney.tif", cv2.IMREAD_GRAYSCALE)
 img = cv2.resize(img, (200, 200))
 img2 = np.copy(img)

 while True:

     lower = cv2.getTrackbarPos("lower", "result")
     upper = cv2.getTrackbarPos("upper", "result")

     for x in range(img.shape[0]):
         for y in range(img.shape[1]):
             img2[x][y] = img[x][y] if (lower < img[x][y] < upper) else 0
     result = cv2.bitwise_and(img, img2)
     cv2.imshow("img", img)
     cv2.imshow("img2", img2)
     cv2.imshow("result", result)
     if cv2.waitKey(1) == ord('q'):
         break
     
cv2.destroyAllWindows()


#%%
# Highlight specific ranges of intesity
 cv2.namedWindow("img2", cv2.WINDOW_KEEPRATIO)

 img = cv2.imread("img/kidney.tif", cv2.IMREAD_GRAYSCALE)
 img2 = np.copy(img)

 threshType = 0
 thresh = 127

 cv2.createTrackbar("threshType", "img2", threshType, 4, doNothing)
 cv2.createTrackbar("thresh", "img2", thresh, 255, doNothing)

 # 0 - THRESH_BINARY
 # 1 - THRESH_BINARY_INV
 # 2 - THRESH_TRUNC
 # 3 - THRESH_TOZERO
 # 4 - THRESH_TOZERO_INV

 while True:
     threshType = cv2.getTrackbarPos("threshType", "img2")
     thresh = cv2.getTrackbarPos("thresh", "img2")

     _, img2 = cv2.threshold(img, thresh, 255, threshType)
     cv2.imshow("img", img)
     cv2.imshow("img2", img2)

     if cv2.waitKey(1) == ord('q'):
         break
cv2.destroyAllWindows()


#%%

# Bit slicing
 cv2.namedWindow("img2", cv2.WINDOW_KEEPRATIO)

 img = cv2.imread("img/dollar.tif", cv2.IMREAD_GRAYSCALE)
 img2 = np.copy(img)

 slice = 7
 cv2.createTrackbar("slice", "img2", slice, 7, doNothing)

 while True:

     slice = cv2.getTrackbarPos("slice", "img2")

     # cv2.bitwise_and(img, 0b00000001, img2) # Using only the four more significant bits.
     # cv2.bitwise_and(img, 0b00000010, img2) # Using only the four more significant bits.
     # cv2.bitwise_and(img, 0b00000100, img2) # Using only the four more significant bits.
     # cv2.bitwise_and(img, 0b00001000, img2) # Using only the four more significant bits.
     # cv2.bitwise_and(img, 0b00010000, img2) # Using only the four more significant bits.
     # cv2.bitwise_and(img, 0b00100000, img2) # Using only the four more significant bits.
     # cv2.bitwise_and(img, 0b01000000, img2) # Using only the four more significant bits.
     # cv2.bitwise_and(img, 0b10000000, img2) # Using only the four more significant bits.

     img2 = cv2.bitwise_and(img, 2 << slice, img2)
     # cv2.bitwise_and(img, 0xf0, img2) # Using only the four more significant bits.
     # cv2.bitwise_and(img, 0xd0, img2) # Using only the three more significant bits.
     # cv2.bitwise_and(img, 0xc0, img2) # Using only the two more significant bits.
     # cv2.bitwise_and(img, 0x80, img2) # Using only the most significant bit.

     img2 = np.asarray(img2, np.float32)
     cv2.normalize(img2, img2, 0, 1, cv2.NORM_MINMAX)
     # img2 = 255 * img2
     # img2.convertTo(img2, CV_8U)
     cv2.imshow("img", img)
     cv2.imshow("img2", img2)
     if cv2.waitKey(1) == ord('q'):
         break
cv2.destroyAllWindows()


#%%

# Histogram equalization
 img = cv2.imread("img/pollen_washedout.tif", cv2.IMREAD_GRAYSCALE)
 img2 = cv2.equalizeHist(img)
 hist = compute_histogram_1C(img)
 hist2 = compute_histogram_1C(img2)
 cv2.imshow("img", img)
 cv2.imshow("img2", img2)
 cv2.imshow("hist", hist)
 cv2.imshow("hist2", hist2)
 while cv2.waitKey(0) != ord('q'):
     pass
cv2.destroyAllWindows()


#%%

# Local Histogram Processing
 img = cv2.imread("img/squares_noisy.tif", cv2.IMREAD_GRAYSCALE)
 img2 = np.zeros(img.shape, img.dtype)
 wsize = 1

 for x in range(wsize, img.shape[0] - wsize):
     for y in range(wsize, img.shape[1] - wsize):
         cv2.equalizeHist(img[y - wsize: y + wsize][x - wsize: x + wsize],
                          img2[y - wsize: y + wsize][x - wsize: x + wsize])

 cv2.imshow("img", img)
 cv2.imshow("img2", img2)

 while cv2.waitKey(1) != ord('q'):
     pass

cv2.destroyAllWindows()


#%%
# Local Histogram Processing
img = cv2.imread('img/tungsten.tif', cv2.IMREAD_GRAYSCALE)

img2 = np.copy(img)

avg_global, std_global = cv2.meanStdDev(img)

avg_global = avg_global[0][0]
std_global = std_global[0][0]

E = 3.0
k0 = 0.4
k1 = 0.02
k2 = 0.4
wsize = 1

for x in range(wsize, img.shape[0] - wsize):
    for y in range(wsize, img.shape[1] - wsize):
        avg_local, std_local = cv2.meanStdDev(img[x-wsize:x+wsize, y-wsize:y+wsize])

        avg_local = avg_local[0][0]
        std_local = std_local[0][0]

        intensity = img[x, y]
        intensity_new = E*intensity if ((avg_local <= k0*avg_global) and (k1*std_global <= std_local) and (std_local <= k2*std_global)) else intensity
        img2[x, y] = intensity_new

cv2.imshow("Original", img)
cv2.imshow("New", img2)

while cv2.waitKey(1) != ord('q'):
    pass

cv2.destroyAllWindows()


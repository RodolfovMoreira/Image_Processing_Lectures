import cv2
import numpy as np
import matplotlib.pyplot as plt


def doNothing(x):
    pass


def createWhiteDisk(height, width, xc, yc, rc):
    disk = np.zeros((height, width), np.float64)

    for x in range(disk.shape[0]):
        for y in range(disk.shape[1]):
            if (x - xc) * (x - xc) + (y - yc) * (y - yc) <= rc * rc:
                disk[x][y] = 1.0
    return disk


def scaleImage2_uchar(src):
    tmp = np.copy(src)
    if src.dtype != np.float32:
        tmp = np.float32(tmp)
    cv2.normalize(tmp, tmp, 1, 0, cv2.NORM_MINMAX)
    tmp = 255 * tmp
    tmp = np.uint8(tmp)
    return tmp


def compute_piecewise_linear_val(val, r1, s1, r2, s2):
    output = 0
    if (0 <= val) and (val <= r1):
        output = (s1 / r1) * val
    if (r1 <= val) and (val <= r2):
        output = ((s2 - s1) / (r2 - r1)) * (val - r1) + s1
    if (r2 <= val) and (val <= 1):
        output = ((1 - s2) / (1 - r2)) * (val - r2) + s2

    return output


def create2DGaussian(rows, cols, mx, my, sx, sy, theta):
    img = np.zeros((rows, cols), np.float32)
    for x in range(rows):
        for y in range(cols):
            img[x, y] = \
                (1.0 / (np.sqrt(2 * np.pi))) * \
                np.exp(- pow((x - mx) * np.cos(theta) - (y - my) * np.sin(theta), 2) / (2 * pow(sx, 2))
                       - pow((x - mx) * np.sin(theta) + (y - my) * np.cos(theta), 2) / (2 * pow(sy, 2)))

    return img


def wait():
    while cv2.waitKey(1) != ord('q'):
        pass

#%%
# Load a color image and visualize each channel separately

 img = cv2.imread("img/baboon.png", cv2.IMREAD_COLOR)
 bgr = cv2.split(img)

 plt.subplot("221"); plt.title("B"); plt.imshow(bgr[0], "gray")
 plt.subplot("222"); plt.title("G"); plt.imshow(bgr[1], "gray")
 plt.subplot("223"); plt.title("R"); plt.imshow(bgr[2], "gray")
 plt.subplot("224"); plt.title("Original"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 plt.show()
        
#%%

# Load a color image and visualize each channel separately

 img = cv2.imread("img/baboon.png", cv2.IMREAD_COLOR)
 bgr = cv2.split(img)
 colormap = cv2.COLORMAP_JET

 plt.subplot("221"); plt.title("B"); plt.imshow(cv2.applyColorMap(bgr[0], colormap))
 plt.subplot("222"); plt.title("G"); plt.imshow(cv2.applyColorMap(bgr[1], colormap))
 plt.subplot("223"); plt.title("R"); plt.imshow(cv2.applyColorMap(bgr[2], colormap))
 plt.subplot("224"); plt.title("Original"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 plt.show()
        
 #%%

# Load a color image and visualize each channel separately

 img = cv2.imread("img/rgbcube_kBKG.png", cv2.IMREAD_COLOR)
 bgr = cv2.split(img)
 colormap = cv2.COLORMAP_JET

 plt.subplot("221"); plt.title("B"); plt.imshow(cv2.applyColorMap(bgr[0], colormap))
 plt.subplot("222"); plt.title("G"); plt.imshow(cv2.applyColorMap(bgr[1], colormap))
 plt.subplot("223"); plt.title("R"); plt.imshow(cv2.applyColorMap(bgr[2], colormap))
 plt.subplot("224"); plt.title("Original"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 plt.show()
        
#%%

# Load a color image and visualize each channel separately

 img = cv2.imread("img/rgbcube_kBKG.png", cv2.IMREAD_COLOR)
 img = 255 - img
 bgr = cv2.split(img)
 colormap = cv2.COLORMAP_JET

 plt.subplot("221"); plt.title("B"); plt.imshow(cv2.applyColorMap(bgr[0], colormap))
 plt.subplot("222"); plt.title("G"); plt.imshow(cv2.applyColorMap(bgr[1], colormap))
 plt.subplot("223"); plt.title("R"); plt.imshow(cv2.applyColorMap(bgr[2], colormap))
 plt.subplot("224"); plt.title("Original"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 plt.show()
        
#%%

# Converting between color spaces - BGR to YRB - Part I
# NTSC colorspace - Part I
 img = cv2.imread("img/baboon.png", cv2.IMREAD_COLOR)

 img2 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
 yrb = cv2.split(img2)

 cv2.imshow("img", img)
 cv2.imshow("img2", img)
 cv2.imshow("y", yrb[0])
 cv2.imshow("Cr", yrb[1])
 cv2.imshow("Cb", yrb[2])

 wait()
        
#%%

# Converting between color spaces - BGR to YRB - Part II
# NTSC colorspace

 img  = cv2.imread("img/baboon.png", cv2.IMREAD_COLOR)
 img2 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 yrb  = cv2.split(img2)
 diff = cv2.absdiff(gray, yrb[0])

 cv2.imshow("img", img)
 cv2.imshow("gray", gray)
 cv2.imshow("diff", diff)
 cv2.imshow("y", yrb[0])
 cv2.imshow("Cr", yrb[1])
 cv2.imshow("Cb", yrb[2])
 wait()
        
#%%

# The HSV colorspace - Part I

 img = cv2.imread("img/rgbcube_kBKG.png", cv2.IMREAD_COLOR)

 img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
 hsv = cv2.split(img2)

 cv2.imshow("img", img)
 cv2.imshow("img2", img2)
 cv2.imshow("h", hsv[0])
 cv2.imshow("s", hsv[1])
 cv2.imshow("v", scaleImage2_uchar(hsv[2]))
 wait()
        
#%%

# The HSV colorspace - Part II

 img = cv2.imread("img/baboon.png", cv2.IMREAD_COLOR)

 img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
 hsv = cv2.split(img2)

 colormap = cv2.COLORMAP_JET

 cv2.imshow("img", img)
 cv2.imshow("img2", img2)
 cv2.imshow("h", hsv[0])
 cv2.imshow("s", hsv[1])
 cv2.imshow("v", hsv[2])
 wait()
        
#%%

# Converting between color spaces - BGR to HSV - Part I
# HSV colorspace

 img = cv2.imread("img/baboon.png", cv2.IMREAD_COLOR)

 img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
 hsv = cv2.split(img2)

 cv2.imshow("img", img)
 cv2.imshow("img2", img2)
 cv2.imshow("h", hsv[0])
 cv2.imshow("s", hsv[1])
 cv2.imshow("v", scaleImage2_uchar(hsv[2]))
 wait()
        
#%%

# Converting between color spaces - BGR to HSV - Part II
# HSV colorspace

 img = cv2.imread("img/chips.png", cv2.IMREAD_COLOR)

 img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
 hsv = cv2.split(img2)

 cv2.imshow("img", img)
 cv2.imshow("img2", img2)
 cv2.imshow("h", hsv[0])
 cv2.imshow("s", hsv[1])
 cv2.imshow("v", scaleImage2_uchar(hsv[2]))
 wait()
        
#%%

# The CMYK colorspace

 img = cv2.imread("img/baboon.png", cv2.IMREAD_COLOR)
 img2 = 255 - img
 ymc = cv2.split(img2)
 colormap = cv2.COLORMAP_JET
 cv2.imshow("img", img)
 cv2.imshow("img2", img2)
 cv2.imshow("y", ymc[0])
 cv2.imshow("m", ymc[1])
 cv2.imshow("c", ymc[2])
 wait()
        
#%%

# Creating BGR color disks

 rows = 1e3
 radius = rows/4
 bx = rows/2
 by = rows/2 - radius/2
 gx = rows/2 - radius/2
 gy = rows/2 + radius/2
 rx = rows/2 + radius/2
 ry = rows/2 + radius/2

 bgr = [
        createWhiteDisk(int(rows), int(rows), int(bx), int(by), int(radius)),
        createWhiteDisk(int(rows), int(rows), int(gx), int(gy), int(radius)),
        createWhiteDisk(int(rows), int(rows), int(rx), int(ry), int(radius))
       ]
 img = cv2.merge(bgr)
 img = scaleImage2_uchar(img)
 cv2.imshow("img", img)
 wait()
        
#%%

# Create CMYK color disks

 rows = 1e3
 radius = rows/4
 bx = rows/2
 by = rows/2 - radius/2
 gx = rows/2 - radius/2
 gy = rows/2 + radius/2
 rx = rows/2 + radius/2
 ry = rows/2 + radius/2

 bgr = [
        createWhiteDisk(int(rows), int(rows), int(bx), int(by), int(radius)),
        createWhiteDisk(int(rows), int(rows), int(gx), int(gy), int(radius)),
        createWhiteDisk(int(rows), int(rows), int(rx), int(ry), int(radius))
       ]
 img = cv2.merge(bgr)
 img = scaleImage2_uchar(img)
 img = 255 - img
 cv2.imshow("img", img)
 wait()
        
#%%

# ****************************************//
# * Intensity transformation of color images.
# *
# * Pice-wise linear transformation of color channels

 img = cv2.imread("img/baboon.png", cv2.IMREAD_COLOR)

 cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)
 cv2.namedWindow("img2", cv2.WINDOW_KEEPRATIO)

 img = cv2.resize(img, (128, 128))

 T0 = 255 * np.ones((256, 256), np.uint8)

 b_x1 = 65
 b_y1 = 65
 b_x2 = 195
 b_y2 = 195

 g_x1 = 65
 g_y1 = 65
 g_x2 = 195
 g_y2 = 195

 r_x1 = 65
 r_y1 = 65
 r_x2 = 195
 r_y2 = 195

 cv2.namedWindow("B_transform", cv2.WINDOW_AUTOSIZE)
 cv2.createTrackbar("b_x1", "B_transform", b_x1, T0.shape[1] - 1, doNothing)
 cv2.createTrackbar("b_y1", "B_transform", b_y1, T0.shape[1] - 1, doNothing)
 cv2.createTrackbar("b_x2", "B_transform", b_x2, T0.shape[1] - 1, doNothing)
 cv2.createTrackbar("b_y2", "B_transform", b_y2, T0.shape[1] - 1, doNothing)

 cv2.namedWindow("G_transform", cv2.WINDOW_AUTOSIZE)
 cv2.createTrackbar("g_x1", "G_transform", g_x1, T0.shape[1] - 1, doNothing)
 cv2.createTrackbar("g_y1", "G_transform", g_y1, T0.shape[1] - 1, doNothing)
 cv2.createTrackbar("g_x2", "G_transform", g_x2, T0.shape[1] - 1, doNothing)
 cv2.createTrackbar("g_y2", "G_transform", g_y2, T0.shape[1] - 1, doNothing)

 cv2.namedWindow("R_transform", cv2.WINDOW_AUTOSIZE)
 cv2.createTrackbar("r_x1", "R_transform", r_x1, T0.shape[1] - 1, doNothing)
 cv2.createTrackbar("r_y1", "R_transform", r_y1, T0.shape[1] - 1, doNothing)
 cv2.createTrackbar("r_x2", "R_transform", r_x2, T0.shape[1] - 1, doNothing)
 cv2.createTrackbar("r_y2", "R_transform", r_y2, T0.shape[1] - 1, doNothing)

 while cv2.waitKey(1) != ord('q'):

     b_x1 = cv2.getTrackbarPos("b_x1", "B_transform")
     b_y1 = cv2.getTrackbarPos("b_y1", "B_transform")
     b_x2 = cv2.getTrackbarPos("b_x2", "B_transform")
     b_y2 = cv2.getTrackbarPos("b_y2", "B_transform")

     g_x1 = cv2.getTrackbarPos("g_x1", "G_transform")
     g_y1 = cv2.getTrackbarPos("g_y1", "G_transform")
     g_x2 = cv2.getTrackbarPos("g_x2", "G_transform")
     g_y2 = cv2.getTrackbarPos("g_y2", "G_transform")

     r_x1 = cv2.getTrackbarPos("r_x1", "R_transform")
     r_y1 = cv2.getTrackbarPos("r_y1", "R_transform")
     r_x2 = cv2.getTrackbarPos("r_x2", "R_transform")
     r_y2 = cv2.getTrackbarPos("r_y2", "R_transform")

     # Draw the transformation function for B channel
     T_B = np.copy(T0)
     p1 = (b_x1, T_B.shape[0] - 1 - b_y1)
     p2 = (b_x2, T_B.shape[0] - 1 - b_y2)
     cv2.line(T_B, (0, T_B.shape[1] - 1), p1, (0,0,0), 2, cv2.LINE_8, 0)
     cv2.circle(T_B, p1, 4, 0, 2, cv2.LINE_8, 0)
     cv2.line(T_B, p1, p2, 0, 2, cv2.LINE_8, 0)
     cv2.circle(T_B, p2, 4, 0, 2, cv2.LINE_8, 0)
     cv2.line(T_B, p2, (T_B.shape[1] - 1, 0), 0, 2, cv2.LINE_8, 0)
     # Draw the transformation function for G channel
     T_G = np.copy(T0)
     p1 = (g_x1, T_G.shape[0] - 1 - g_y1)
     p2 = (g_x2, T_G.shape[0] - 1 - g_y2)
     cv2.line(T_G, (0, T_G.shape[1] - 1), p1, (0,0,0), 2, cv2.LINE_8, 0)
     cv2.circle(T_G, p1, 4, 0, 2, cv2.LINE_8, 0)
     cv2.line(T_G, p1, p2, 0, 2, cv2.LINE_8, 0)
     cv2.circle(T_G, p2, 4, 0, 2, cv2.LINE_8, 0)
     cv2.line(T_G, p2, (T_G.shape[1] - 1, 0), 0, 2, cv2.LINE_8, 0)
     # Draw the transformation function for R channel
     T_R = np.copy(T0)
     p1 = (r_x1, T_R.shape[0] - 1 - r_y1)
     p2 = (r_x2, T_R.shape[0] - 1 - r_y2)
     cv2.line(T_R, (0, T_R.shape[1] - 1), p1, (0,0,0), 2, cv2.LINE_8, 0)
     cv2.circle(T_R, p1, 4, 0, 2, cv2.LINE_8, 0)
     cv2.line(T_R, p1, p2, 0, 2, cv2.LINE_8, 0)
     cv2.circle(T_R, p2, 4, 0, 2, cv2.LINE_8, 0)
     cv2.line(T_R, p2, (T_R.shape[1] - 1, 0), 0, 2, cv2.LINE_8, 0)

     # Clone the original image
     img2 = np.copy(img)

     # Split its channels
     bgr = cv2.split(img2)
     B = bgr[0]
     G = bgr[1]
     R = bgr[2]

     for x in range(0, img2.shape[1]):
         for y in range(0, img2.shape[0]):
             B[x, y] = 255 * compute_piecewise_linear_val(B[x, y] / 255.0,
                                                                   (b_x1 / 255.0),
                                                                   (b_y1 / 255.0),
                                                                   (b_x2 / 255.0),
                                                                   (b_y2 / 255.0))

             G[x, y] = 255 * compute_piecewise_linear_val(G[x, y] / 255.0,
                                                                   (g_x1 / 255.0),
                                                                   (g_y1 / 255.0),
                                                                   (g_x2 / 255.0),
                                                                   (g_y2 / 255.0))

             R[x, y] = 255 * compute_piecewise_linear_val(R[x, y] / 255.0,
                                                                   (r_x1 / 255.0),
                                                                   (r_y1 / 255.0),
                                                                   (r_x2 / 255.0),
                                                                   (r_y2 / 255.0))

     img2 = cv2.merge(bgr)

     cv2.imshow("img", img)
     cv2.imshow("img2", img2)
     cv2.imshow("B_transform", T_B)
     cv2.imshow("G_transform", T_G)
     cv2.imshow("R_transform", T_R)
        
#%%

# Spatial transformation - Blurring

 img = cv2.imread("img/baboon.png", cv2.IMREAD_COLOR)
 wsize = 3

 cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)
 cv2.namedWindow("img2", cv2.WINDOW_KEEPRATIO)

 cv2.createTrackbar("wsize", "img2", wsize, 50, doNothing)

 while cv2.waitKey(1) != ord('q'):
     wsize = cv2.getTrackbarPos("wsize", "img2")

     img2 = cv2.blur(img, (wsize + 1, wsize + 1))
     cv2.imshow("img", img)
     cv2.imshow("img2", img2)
        
#%%

# Spatial transformation - Sharpening

 img = cv2.imread("img/baboon.png", cv2.IMREAD_COLOR)
 wsize = 3

 cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)
 cv2.namedWindow("img2", cv2.WINDOW_KEEPRATIO)

 cv2.createTrackbar("wsize", "img2", wsize, 10, doNothing)

 while cv2.waitKey(1) != ord('q'):
     wsize = cv2.getTrackbarPos("wsize", "img2")

     img2 = cv2.Laplacian(img, cv2.CV_16S, ksize=2*wsize + 1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

     print(img2)

     cv2.imshow("img", img)
     cv2.imshow("img2", img2)
        
#%%

# Working directly in BGR colorspace
# Using a prepared image

 rows = 100
 cols = 200

 R = np.zeros((rows, cols), np.float32)
 B = np.copy(R);
 for col in range(int(cols / 2), cols):
     for row in range(0, rows):
         R[row, col] = 1

 G = np.copy(R)

 for col in range(0, cols):
     for row in range(0, int(rows / 2)):
         B[row, col] = 1

 bgr = [B, G, R]
 img = cv2.merge(bgr)

 # Using the "baboon" image
 # img = cv2.imread("img/baboon.png", cv2.IMREAD_COLOR)
 # img = np.float32(img)
 # planes = cv2.split(img)
 # B = planes[0]
 # G = planes[1]
 # R = planes[2]

 dBx = cv2.Sobel(B, cv2.CV_32F, 1, 0, 3, 1, 0, cv2.BORDER_DEFAULT)
 dBy = cv2.Sobel(B, cv2.CV_32F, 0, 1, 3, 1, 0, cv2.BORDER_DEFAULT)
 dGx = cv2.Sobel(G, cv2.CV_32F, 1, 0, 3, 1, 0, cv2.BORDER_DEFAULT)
 dGy = cv2.Sobel(G, cv2.CV_32F, 0, 1, 3, 1, 0, cv2.BORDER_DEFAULT)
 dRx = cv2.Sobel(R, cv2.CV_32F, 1, 0, 3, 1, 0, cv2.BORDER_DEFAULT)
 dRy = cv2.Sobel(R, cv2.CV_32F, 0, 1, 3, 1, 0, cv2.BORDER_DEFAULT)

 Gxx = np.multiply(dBx, dBx) + np.multiply(dGx, dGx) + np.multiply(dRx, dRx)
 Gyy = np.multiply(dBy, dBy) + np.multiply(dGy, dGy) + np.multiply(dRy, dRy)
 Gxy = np.multiply(dBx, dBy) + np.multiply(dGx, dGy) + np.multiply(dRx, dRy)

 Theta = np.zeros(img.shape, Gxx.dtype)
 F = np.zeros(img.shape, Gxx.dtype)

 for x in range(img.shape[0]):
     for y in range(img.shape[1]):
         gxx = Gxx[x, y]
         gyy = Gyy[x, y]
         gxy = Gxy[x, y]
         theta = 0.5 * np.arctan2((2 * gxy), (gxx - gyy))
         Theta[x, y] = theta
         F[x, y] = np.sqrt(0.5 * (gxx + gyy + (gxx + gyy) * np.cos(2 * theta) + 2 * gxy * np.sin(2 * theta)))

 cv2.imshow("img", scaleImage2_uchar(img))
 cv2.imshow("F", scaleImage2_uchar(F))
 wait()
        
#%%

# Image Segmentation in the BGR colorspace - Part I

 img = cv2.imread("img/baboon.png", cv2.IMREAD_COLOR)

 sp = 10
 sr = 100
 maxLevel = 1

 img2 = cv2.pyrMeanShiftFiltering(img, sp, sr, maxLevel, cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS)

 cv2.imshow("img", img)
 cv2.imshow("img2", img2)
 wait()
        
#%%

# Creating a gaussian image

 img = np.zeros((100, 100), np.float32)
 mx = 50
 my = 50
 sx = 20
 sy = 10

 colormap = cv2.COLORMAP_JET
 theta_slider = 20

 cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)

 cv2.createTrackbar("mx", "img", mx, img.shape[1], doNothing)
 cv2.createTrackbar("my", "img", my, img.shape[1], doNothing)
 cv2.createTrackbar("sx", "img", sx, img.shape[1], doNothing)
 cv2.createTrackbar("sy", "img", sy, img.shape[1], doNothing)
 cv2.createTrackbar("theta", "img", theta_slider, 100, doNothing)
 cv2.createTrackbar("colormap", "img", colormap, 7, doNothing)

 while cv2.waitKey(1) != ord('q'):
     mx = cv2.getTrackbarPos("mx", "img")
     my = cv2.getTrackbarPos("my", "img")
     sx = cv2.getTrackbarPos("sx", "img")
     sy = cv2.getTrackbarPos("sy", "img")
     theta_slider = cv2.getTrackbarPos("theta", "img")
     colormap = cv2.getTrackbarPos("colormap", "img")

     theta = theta_slider * 2 * np.pi / 100
     img = create2DGaussian(100, 100, mx, my, sx, sy, theta)
     cv2.imshow("img", cv2.applyColorMap(scaleImage2_uchar(img), colormap))

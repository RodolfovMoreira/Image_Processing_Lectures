import cv2
import numpy as np
import matplotlib.pyplot as plt

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
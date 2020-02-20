import cv2
import numpy as np
import matplotlib.pyplot as plt

cv2.__version__

folder = "home/rodol/Documentos/4-PROCESSAMENTO_DE_IMAGENS-8-/AB2/Questões_de_revisão"


def do_nothing(x):
    pass


def create2DGaussian(rows=100,
                     cols=100,
                     mx=50,
                     my=50,
                     sx=10,
                     sy=100,
                     theta=0):
    xx0, yy0 = np.meshgrid(range(rows), range(cols))
    xx0 -= mx
    yy0 -= my
    theta = np.deg2rad(theta)
    xx = xx0 * np.cos(theta) - yy0 * np.sin(theta)
    yy = xx0 * np.sin(theta) + yy0 * np.cos(theta)
    try:
        img = np.exp(- ((xx ** 2) / (2 * sx ** 2) +
                        (yy ** 2) / (2 * sy ** 2)))
    except ZeroDivisionError:
        img = np.zeros((rows, cols), dtype='float64')

    cv2.normalize(img, img, 1, 0, cv2.NORM_MINMAX)
    return img


def reconstruct(seeds, mask):
    last_image = 0

    while True:
        seeds = cv2.dilate(seeds, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        seeds = cv2.bitwise_and(seeds, mask)

        if (seeds == last_image).all():
            break

        last_image = seeds

    return seeds


def color_thresholding(image, min_val_hsv, max_val_hsv):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, min_val_hsv, max_val_hsv)
    imask = mask > 0
    extracted = np.zeros_like(image, image.dtype)
    extracted[imask] = image[imask]
    return extracted


# %%  Primeira Questao
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

folder = "home/rodol/Documentos/4-PROCESSAMENTO_DE_IMAGENS-8-/AB2/Questões_de_revisão"
#
#img = cv2.imread(folder + '1.png', cv2.IMREAD_GRAYSCALE)

img = cv2.imread(os.path.join(folder,'1.png'), cv2.IMREAD_GRAYSCALE)

opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)

result = img - opened

cv2.imshow("Original", img)
cv2.imshow("Opened", opened)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%  Segunda Questao

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

folder = "home/rodol/Documentos/4-PROCESSAMENTO_DE_IMAGENS-8-/AB2/Questões_de_revisão"
img = cv2.imread(os.path.join(folder,'1.png'), cv2.IMREAD_GRAYSCALE)

opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)

thresholded = cv2.adaptiveThreshold(img, 60, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 0)
opened = cv2.morphologyEx(thresholded, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                          iterations=4)
only_balls = reconstruct(opened, thresholded)

filter = np.zeros(img.shape, img.dtype)
percent = 0.55
filter[:, int(filter.shape[1] * percent):filter.shape[1]] = 1

filtered = opened * filter

large_balls = reconstruct(filtered, thresholded)
tiny_balls = only_balls - large_balls

cv2.normalize(large_balls, large_balls, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(tiny_balls, tiny_balls, 0, 255, cv2.NORM_MINMAX)

cv2.imshow("Original", img)
cv2.imshow("Large Balls", large_balls)
cv2.imshow("Tiny Balls", tiny_balls)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %% Terceira Questao
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

background = np.zeros((400, 400), dtype=np.float32)

radius = int(background.shape[0] * 0.7)
center = (int(background.shape[0] / 2), int(background.shape[1] / 2))

v = np.copy(background)
disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
v[int(center[0] - radius / 2):int(center[0] + radius / 2),
int(center[1] - radius / 2):int(center[1] + radius / 2)] = disk

s = np.float32(create2DGaussian(background.shape[0], background.shape[1], center[0], center[1], 90, 90))
s = v * s

h = np.copy(background)

for theta in np.arange(0, np.pi * 2, np.pi * 2 / 300):
    for r in range(0, int(radius / 2)):
        h[int(center[0] - r * np.sin(theta))][int(center[1] - r * np.cos(theta))] = theta / 9

h = cv2.morphologyEx(h, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

h = np.uint8(255 * h)
s = np.uint8(255 * s)
v = np.uint8(255 * v)

cv2.imshow("H", h)
cv2.imshow("S", s)
cv2.imshow("V", v)

hsv = cv2.merge((h, s, v))
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow("BGR", bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %% Quarta Questao
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

folder = "home/rodol/Documentos/4-PROCESSAMENTO_DE_IMAGENS-8-/AB2/Questões_de_revisão"
img = cv2.imread(os.path.join(folder,'4.png'), cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 160)

for line in lines:
    rho = line[0][0]
    theta = line[0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 4)

cv2.imshow("Rubik", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %% Quinta Questao
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_thresholding(image, min_val_hsv, max_val_hsv):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, min_val_hsv, max_val_hsv)
    imask = mask > 0
    extracted = np.zeros_like(image, image.dtype)
    extracted[imask] = image[imask]
    return extracted

folder = "home/rodol/Documentos/4-PROCESSAMENTO_DE_IMAGENS-8-/AB2/Questões_de_revisão"
img = cv2.imread(os.path.join(folder,'5.png'))

bgr = color_thresholding(img, (128, 0, 0), (166, 255, 255))
cv2.imshow("Orignal", img)
cv2.imshow("Only Vegetables", img - bgr)
cv2.waitKey(0)

cv2.destroyAllWindows()


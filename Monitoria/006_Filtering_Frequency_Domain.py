# %% Global definitions
folder = '/home/bruno/Documentos/Workspaces/Python_Projects/Aulas_PDI/img/'

# %% Import libraries
import cv2
import numpy as np


# %% Define functions
def doNothing(x):
    pass


def createWhiteDisk(height=100, width=100, xc=50, yc=50, rc=20):
    disk = np.zeros((height, width), np.float64)
    for x in range(disk.shape[0]):
        for y in range(disk.shape[1]):
            if (x - xc) * (x - xc) + (y - yc) * (y - yc) <= rc * rc:
                disk[x][y] = 1.0
    return disk


def createWhiteDisk2(height=100, width=100, xc=50, yc=50, rc=20):
    xx, yy = np.meshgrid(range(height), range(width))
    img = np.array(
        ((xx - xc) ** 2 + (yy - yc) ** 2 - rc ** 2) < 0).astype('float64')
    return img


def scaleImage2_uchar(src):
    tmp = np.copy(src)
    if src.dtype != np.float32:
        tmp = np.float32(tmp)
    cv2.normalize(tmp, tmp, 1, 0, cv2.NORM_MINMAX)
    tmp = 255 * tmp
    tmp = np.uint8(tmp)
    return tmp


def createCosineImage(height, width, freq, theta):
    img = np.zeros((height, width), dtype=np.float64)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img[x][y] = np.cos(
                2 * np.pi * freq * (x * np.cos(theta) - y * np.sin(theta)))
    return img


def createCosineImage2(height, width, freq, theta):
    img = np.zeros((height, width), dtype=np.float64)
    xx, yy = np.meshgrid(range(height), range(width))
    theta = np.deg2rad(theta)
    rho = (xx * np.cos(theta) - yy * np.sin(theta))
    img[:] = np.cos(2 * np.pi * freq * rho)
    return img


def applyLogTransform(img):
    img2 = np.copy(img)
    img2 += 1
    img2 = np.log(img2)
    return img2


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


# %% Demonstrate the creation of a Gaussian filter
rows = 400
cols = 400
theta = 0
xc = 200
yc = 200
sx = 120
sy = 40

cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar("xc", "img", xc, int(rows), doNothing)
cv2.createTrackbar("yc", "img", yc, int(cols), doNothing)
cv2.createTrackbar("sx", "img", sx, int(rows), doNothing)
cv2.createTrackbar("sy", "img", sy, int(cols), doNothing)
cv2.createTrackbar("theta", "img", theta, 360, doNothing)
while 0xFF & cv2.waitKey(1) != ord('q'):
    xc = cv2.getTrackbarPos("xc", "img")
    yc = cv2.getTrackbarPos("yc", "img")
    sx = cv2.getTrackbarPos("sx", "img")
    sy = cv2.getTrackbarPos("sy", "img")
    theta = cv2.getTrackbarPos("theta", "img")
    img = create2DGaussian(rows, cols, xc, yc, sx, sy, theta)
    cv2.imshow('img', cv2.applyColorMap(scaleImage2_uchar(img),
                                        cv2.COLORMAP_JET))
cv2.destroyAllWindows()

# %% The Discrete Fourier Transform - Part I - Obtaining real and imaginary
# parts of the Fourier Transform
img = cv2.imread(folder + 'pollen.jpg', cv2.IMREAD_GRAYSCALE)

cv2.namedWindow("Original", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Plane 0 - Real", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Plane 1 - Imaginary", cv2.WINDOW_KEEPRATIO)

planes = [np.zeros(img.shape, dtype=np.float64),
          np.zeros(img.shape, dtype=np.float64)]
planes[0][:] = np.float64(img[:])

img2 = cv2.merge(planes)
img2 = cv2.dft(img2)

planes = cv2.split(img2)

# cv2.normalize(planes[0], planes[0], 1, 0, cv2.NORM_MINMAX)
# cv2.normalize(planes[1], planes[1], 1, 0, cv2.NORM_MINMAX)

while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow('Original', img)
    cv2.imshow('Plane 0 - Real', planes[0])
    cv2.imshow('Plane 1 - Imaginary', planes[1])
cv2.destroyAllWindows()

# %% DFT - Part II -> Applying the log transform
img = cv2.imread(folder + 'rectangle.jpg', cv2.IMREAD_GRAYSCALE)

cv2.namedWindow("Original", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Plane 0 - Real", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Plane 1 - Imaginary", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Mag", cv2.WINDOW_KEEPRATIO)

planes = [np.zeros(img.shape, dtype=np.float64),
          np.zeros(img.shape, dtype=np.float64)]

planes[0][:] = np.float64(img[:])
planes[1][:] = np.float64(img[:])

cv2.normalize(planes[0], planes[0], 1, 0, cv2.NORM_MINMAX)
cv2.normalize(planes[1], planes[1], 1, 0, cv2.NORM_MINMAX)

img2 = cv2.merge(planes)
img2 = cv2.dft(img2)
planes = cv2.split(img2)

mag = cv2.magnitude(planes[0], planes[1])
mag += 1
mag = np.log(mag)

cv2.normalize(mag, mag, 1, 0, cv2.NORM_MINMAX)

while cv2.waitKey(1) != ord('q'):
    cv2.imshow('Original', img)
    cv2.imshow('Plane 0 - Real', planes[0])
    cv2.imshow('Plane 1 - Imaginary', planes[1])
    cv2.imshow('Mag', mag)
cv2.destroyAllWindows()

# %% DFT - Part III -> Shifting the Transform
img = cv2.imread(folder + 'lena.png', cv2.IMREAD_GRAYSCALE)

cv2.namedWindow("Original", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Mag", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Mag Shifted", cv2.WINDOW_KEEPRATIO)

planes = [np.zeros(img.shape, dtype=np.float64),
          np.zeros(img.shape, dtype=np.float64)]

planes[0][:] = np.float64(img[:])
planes[1][:] = np.float64(img[:])
cv2.normalize(planes[0], planes[0], 1, 0, cv2.NORM_MINMAX)
cv2.normalize(planes[1], planes[1], 1, 0, cv2.NORM_MINMAX)

img2 = cv2.merge(planes)
img2 = cv2.dft(img2)
planes = cv2.split(img2)

mag = cv2.magnitude(planes[0], planes[1])
mag += 1
mag = np.log(mag)

cv2.normalize(mag, mag, 1, 0, cv2.NORM_MINMAX)

while cv2.waitKey(1) != ord('q'):
    # print(mag)
    cv2.imshow('Original', img)
    cv2.imshow('Mag Shifted', np.fft.fftshift(mag))
    cv2.imshow('Mag', mag)
cv2.destroyAllWindows()

# %% The Discrete Fourier Transform
rows = 200
cols = 200
disk = np.zeros((rows, cols), np.float32)

cv2.namedWindow("disk", cv2.WINDOW_KEEPRATIO)

xc = 100
yc = 100
radius = 20

cv2.createTrackbar("xc", "disk", xc, disk.shape[0], doNothing)
cv2.createTrackbar("yc", "disk", yc, disk.shape[1], doNothing)
cv2.createTrackbar("radius", "disk", radius, int(disk.shape[1] / 2), doNothing)

while cv2.waitKey(1) != ord('q'):
    xc = cv2.getTrackbarPos("xc", "disk")
    yc = cv2.getTrackbarPos("yc", "disk")
    radius = cv2.getTrackbarPos("radius", "disk")
    disk = createWhiteDisk2(200, 200, xc, yc, radius)

    cv2.imshow("disk", disk)
cv2.destroyAllWindows()

# %% The Discrete Fourier Transform - Part III - Lowpass Filtering
# Ressaltar o surgimento de "falseamento", isto é, frequencia notáveis
# quando é feita a transformada inversa. Isto ocorre pq o filtro é IDEAL.
# Comparar o resultado da filtragem usando uma Gaussiana como filtro.
img = cv2.imread(folder + "lena.png", cv2.IMREAD_GRAYSCALE)

radius = 50
cv2.namedWindow("mask", cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar("radius", "mask", radius, img.shape[0], doNothing)

while cv2.waitKey(1) != ord('q'):
    radius = cv2.getTrackbarPos("radius", "mask")

    #    mask = createWhiteDisk2(img.shape[0],
    #                            img.shape[1],
    #                            int(img.shape[0] / 2),
    #                            int(img.shape[1] / 2),
    #                            radius)
    mask = create2DGaussian(img.shape[0],
                            img.shape[1],
                            int(img.shape[0] / 2),
                            int(img.shape[1] / 2),
                            radius,
                            radius,
                            theta=0)

    img = np.float32(img)

    planes = [img, np.zeros(img.shape, dtype=np.float32)]

    img2 = cv2.merge(planes)
    img2 = cv2.dft(img2)
    planes = cv2.split(img2)

    planes[0] = np.multiply(np.fft.fftshift(mask), planes[0])
    planes[1] = np.multiply(np.fft.fftshift(mask), planes[1])
    img2 = cv2.merge(planes)
    img2 = cv2.idft(img2)
    img2 = np.fft.fftshift(img2)

    cv2.imshow("img", scaleImage2_uchar(img))
    cv2.imshow("planes_0", np.fft.fftshift(planes[0]))
    cv2.imshow("planes_1", np.fft.fftshift(planes[1]))
    cv2.imshow("mask", np.fft.fftshift(mask))
    cv2.imshow("img2", np.fft.fftshift(scaleImage2_uchar(img2[:, :, 1])))
cv2.destroyAllWindows()

# %% The Discrete Fourier Transform - Part IV - Highpass Filtering
img = cv2.imread(folder + "lena.png", cv2.IMREAD_GRAYSCALE)
radius = 50
cv2.namedWindow("mask", cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar("radius", "mask", radius, img.shape[0], doNothing)

while cv2.waitKey(1) != ord('q'):
    radius = cv2.getTrackbarPos("radius", "mask")

    #    mask = createWhiteDisk2(img.shape[0],
    #                            img.shape[1],
    #                            int(img.shape[0] / 2),
    #                            int(img.shape[1] / 2),
    #                            radius)
    mask = 1.0 - create2DGaussian(img.shape[0],
                                  img.shape[1],
                                  int(img.shape[0] / 2),
                                  int(img.shape[1] / 2),
                                  radius + 1,
                                  radius + 1,
                                  theta=0)

    img = np.float32(img)

    planes = [img, np.zeros(img.shape, dtype=np.float32)]

    img2 = cv2.merge(planes)
    img2 = cv2.dft(img2)
    planes = cv2.split(img2)

    planes[0] = np.multiply(np.fft.fftshift(mask), planes[0])
    planes[1] = np.multiply(np.fft.fftshift(mask), planes[1])
    img2 = cv2.merge(planes)
    img2 = cv2.idft(img2)
    img2 = np.fft.fftshift(img2)

    cv2.imshow("img", scaleImage2_uchar(img))
    cv2.imshow("planes_0", np.fft.fftshift(planes[0]))
    cv2.imshow("planes_1", np.fft.fftshift(planes[1]))
    cv2.imshow("mask", mask)
    cv2.imshow("img2", np.fft.fftshift(scaleImage2_uchar(img2[:, :, 1])))
cv2.destroyAllWindows()

# %% The Discrete Fourier Transform - Visualizing sinusoidal images - Part I
rows = 250
cols = 250
freq = 1
theta = 0

cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar("Freq", "img", freq, 500, doNothing)
cv2.createTrackbar("Theta", "img", theta, 360, doNothing)

while cv2.waitKey(1) != ord('q'):
    freq = cv2.getTrackbarPos("Freq", "img")
    theta = cv2.getTrackbarPos("Theta", "img")

    #    img = createCosineImage(rows,
    #                            cols,
    #                            float(freq/1e3),
    #                            float(2 * np.pi * theta/100.0)) # ~0.2 second
    img = createCosineImage2(rows,
                             cols,
                             float(freq / 1e3),
                             theta)  # ~0.001 second
    cv2.imshow("img", scaleImage2_uchar(img))
cv2.destroyAllWindows()

# %% The Discrete Fourier Transform - Visualizing sinusoidal images - Part II
rows = 250
cols = 250
freq = 1
theta = 2

cv2.namedWindow("mag", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar("Freq", "img", freq, 500, doNothing)
cv2.createTrackbar("Theta", "img", theta, 100, doNothing)

while cv2.waitKey(1) != ord('q'):
    freq = cv2.getTrackbarPos("Freq", "img")
    theta = cv2.getTrackbarPos("Theta", "img")

    img = createCosineImage2(rows, cols, float(freq / 1e3), theta)
    img3 = np.copy(img)
    planes = [img3, np.zeros(img3.shape, np.float64)]
    img2 = cv2.merge(planes)
    img2 = cv2.dft(img2)
    planes = cv2.split(img2)
    mag = cv2.magnitude(planes[0], planes[1])
    mag = applyLogTransform(mag)

    cv2.imshow("img", cv2.applyColorMap(scaleImage2_uchar(img),
                                        cv2.COLORMAP_JET))
    cv2.imshow("mag", cv2.applyColorMap(np.fft.fftshift(scaleImage2_uchar(mag)),
                                        cv2.COLORMAP_JET))
cv2.destroyAllWindows()

# %% The Discrete Fourier Transform - Adding sinusoidal noise to images - Part I
cv2.namedWindow("mag", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)

img = cv2.imread(folder + "lena.png", cv2.IMREAD_GRAYSCALE)

img = np.float32(img)
img = img / 255.0

rows = img.shape[0]
cols = img.shape[1]

freq = 90
theta = 10
gain = 30

cv2.createTrackbar("Freq", "img", freq, 500, doNothing)
cv2.createTrackbar("Theta", "img", theta, 100, doNothing)
cv2.createTrackbar("Gain", "img", gain, 100, doNothing)

while cv2.waitKey(1) != ord('q'):
    freq = cv2.getTrackbarPos("Freq", "img")
    theta = cv2.getTrackbarPos("Theta", "img")
    gain = cv2.getTrackbarPos("Gain", "img")

    noise = createCosineImage2(rows, cols, float(freq / 1e3), theta)
    noise = img + float(gain / 100.0) * noise

    img3 = np.copy(noise)
    planes = [img3, np.zeros(img3.shape, np.float64)]
    img2 = cv2.merge(planes)
    img2 = cv2.dft(img2)
    planes = cv2.split(img2)
    mag = cv2.magnitude(planes[0], planes[1])
    mag = applyLogTransform(mag)

    cv2.imshow("img", scaleImage2_uchar(noise))
    cv2.imshow("mag", cv2.applyColorMap(
        np.fft.fftshift(scaleImage2_uchar(mag)),
        cv2.COLORMAP_OCEAN))
cv2.destroyAllWindows()

# %% The Discrete Fourier Transform - Adding sinusoidal noise to images - Part II
cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("mask", cv2.WINDOW_KEEPRATIO)

img = cv2.imread(folder + "lena.png", cv2.IMREAD_GRAYSCALE)
img = np.float32(img)
img = img / 255.0;

rows = img.shape[0]
cols = img.shape[1]

freq = 90
theta = 10
gain = 30

cv2.createTrackbar("Freq", "img", freq, 500, doNothing)
cv2.createTrackbar("Theta", "img", theta, 100, doNothing)
cv2.createTrackbar("Gain", "img", gain, 100, doNothing)

bandwidth = 2
outer_radius = 256 - 210 + bandwidth
inner_radius = 256 - 210 - bandwidth
cv2.createTrackbar("in_radius", "mask", inner_radius, img.shape[1], doNothing)
cv2.createTrackbar("out_radius", "mask", outer_radius, img.shape[1], doNothing)

while cv2.waitKey(1) != ord('q'):
    freq = cv2.getTrackbarPos("Freq", "img")
    theta = cv2.getTrackbarPos("Theta", "img")
    gain = cv2.getTrackbarPos("Gain", "img")

    outer_radius = cv2.getTrackbarPos("in_radius", "mask")
    inner_radius = cv2.getTrackbarPos("out_radius", "mask")

    noise = img + float(gain / 100.0) * createCosineImage2(
        rows, cols, float(freq / 1e3), theta)

    mask = 1 - (createWhiteDisk2(rows, cols, int(cols / 2),
                                 int(rows / 2), outer_radius) - createWhiteDisk2(rows, cols, int(cols / 2),
                                                                                 int(rows / 2), inner_radius))

    planes = [np.copy(noise), np.zeros(noise.shape, np.float64)]
    img2 = cv2.merge(planes)
    img2 = cv2.dft(img2)
    planes = cv2.split(img2)
    mag = cv2.magnitude(planes[0], planes[1])
    mag = applyLogTransform(mag)
    planes[0] = np.multiply(np.fft.fftshift(mask), planes[0])
    planes[1] = np.multiply(np.fft.fftshift(mask), planes[1])
    tmp = cv2.merge(planes)
    tmp = cv2.idft(tmp)

    cv2.imshow("img", scaleImage2_uchar(noise))
    cv2.imshow("mag", cv2.applyColorMap(np.fft.fftshift(scaleImage2_uchar(mag)), cv2.COLORMAP_OCEAN))
    cv2.imshow("mask", scaleImage2_uchar(mask))
    cv2.imshow("tmp", scaleImage2_uchar(tmp[:, :, 0]))
cv2.destroyAllWindows()
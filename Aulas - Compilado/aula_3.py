#%% IMPORT DAS BIBLIOTECAS
import cv2
import numpy as np

#%% CARREGAR AS IMAGENS
img1 = cv2.imread("lena.png", cv2.IMREAD_COLOR)
img1_float = np.float32(img1)
img2 = cv2.imread("baboon.png", cv2.IMREAD_COLOR)
img2_float = np.float32(img2)

#%% SOMA DE DUAS IMAGENS MULTIPLICADAS POR UM ESCALAR
img_result = np.uint8(0.6 * img1_float + 0.4 * img2_float)
cv2.namedWindow("Soma", cv2.WINDOW_KEEPRATIO)
cv2.imshow("Soma", img_result)

#%% OPERADOR MAXIMO
img_max = cv2.max(img1, img2)
cv2.namedWindow("Maximo", cv2.WINDOW_KEEPRATIO)
cv2.imshow("Maximo", img_max)

#%% DIFERENCA ABSOLUTA
img_diff = cv2.absdiff(img1, img2)
cv2.namedWindow("Diferenca absoluta", cv2.WINDOW_KEEPRATIO)
cv2.imshow("Diferenca absoluta", img_diff)

#%% TRANSFORMACOES GEOMETRICAS
img_stretched1 = cv2.resize(img1, None, fx = 2, fy = 1, interpolation = cv2.INTER_CUBIC)
cv2.namedWindow("Esticada Horizontal", cv2.WINDOW_KEEPRATIO)
cv2.imshow("Esticada Horizontal", img_stretched1)

img_stretched2 = cv2.resize(img1, None, fx = 1, fy = 2, interpolation = cv2.INTER_CUBIC)
cv2.namedWindow("Esticada Vertical", cv2.WINDOW_KEEPRATIO)
cv2.imshow("Esticada Vertical", img_stretched2)

M2 = cv2.getRotationMatrix2D(((510-1)/2.0,(510-1)/2.0),90,1)
img_rotated = cv2.warpAffine(img1,M2,(510,510))
cv2.namedWindow("Rotacionada", cv2.WINDOW_KEEPRATIO)
cv2.imshow("Rotacionada", img_rotated)

img_resized = cv2.resize(img1, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
cv2.namedWindow("Redimensionada", cv2.WINDOW_KEEPRATIO)
cv2.imshow("Redimensionada", img_resized)

#%% FECHAR TODAS AS JANELAS
cv2.waitKey(0)
cv2.destroyAllWindows()
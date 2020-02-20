#%% IMPORT DAS BIBLIOTECAS
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%% IMAGEM ORIGINAL
img = cv2.imread("raio.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("normal", img)
plt.subplot("311")
plt.hist(img.ravel(), 256, [0, 255])

#%% EQUALIZACAO DO HISTOGRAMA
img_eq = cv2.equalizeHist(img)
cv2.imshow("equalized", img_eq)
plt.subplot("312")
plt.hist(img_eq.ravel(), 256, [0, 255])

#%% TRANSFORMACAO LOG
#img_log = cv2.log(img)
#cv2.imshow("log", img_log)

#%% TRANSFORMACAO GAMMA
inv_gamma = 1.0 / 2.5
table = np.array([((img / 255.0) ** inv_gamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

#%% FECHAR TODAS AS JANELAS
cv2.waitKey(0)
cv2.destroyAllWindows()
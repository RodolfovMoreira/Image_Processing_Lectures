#%% IMPORTANDO AS BIBLIOTECAS
import cv2
import numpy as np

#%% CRIANDO AS IMAGENS
img1 = cv2.imread("baboon.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

#%% DIFERENCIACAO DE IMAGENS
diff_img = cv2.absdiff(img1, img2)
cv2.imshow("Diference", diff_img)

#%% OPERACOES LOGICAS COM IMAGENS
and_img = cv2.bitwise_and(img1, img2) # AND
cv2.imshow("And", and_img)
or_img = cv2.bitwise_or(img1, img2) # OR
cv2.imshow("Or", or_img)
not_img = cv2.bitwise_not(img1) # NOT
cv2.imshow("Not", not_img)
xor_img = cv2.bitwise_xor(img1, img2) # XOR
cv2.imshow("Xor", xor_img)

#%% UNIAO
union_img = cv2.max(img1, img2)
cv2.imshow("Union", union_img)

#%% FECHAR TODAS AS JANELAS
cv2.waitKey(0)
cv2.destroyAllWindows()
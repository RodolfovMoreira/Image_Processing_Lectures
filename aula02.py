# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 09:57:48 2018

@author: alunoic
"""
# Mulitplicando cada imagem por uma cte. e somando-as

import numpy as np
import cv2
import matplotlib.pyplot as plt

#cv2.__version__
cte = float(0.5)
img1 = cv2.imread('/home/alunoic/Documentos/lena.png', cv2.IMREAD_COLOR)# Carregando a imagem 1
img1 = cv2.resize(img1, (200,200))# Redimensionando a imagem
img1 = cte * img1# Multiplicando a imagem pela cte
#cv2.imshow('img1', img1)
img2 = cv2.imread('/home/alunoic/Documentos/baboon.png', cv2.IMREAD_COLOR)# Carregando a imagem 2
img2 = cv2.resize(img2, (200,200))# Redimensionando a imagem 2
img2 = cte * img2# Multiplicando a imagem pela cte

img_sum = img1 + img2# Somando as imagens
np.where(img_sum > 255, 255, img_sum)

img_sum = img_sum/255# Normalizando

img_sum_result = cv2.imshow('Imagens somadas', img_sum)
cv2.waitKey(0)

#%%
# Mulitplicando cada imagem por seu valor maximo e somando-as
import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('/home/alunoic/Documentos/lena.png', cv2.IMREAD_COLOR)# Carregando a imagem 1
img1 = cv2.resize(img1, (200,200))# Redimensionando a imagem

img2 = cv2.imread('/home/alunoic/Documentos/baboon.png', cv2.IMREAD_COLOR)# Carregando a imagem 2
img2 = cv2.resize(img2, (200,200))# Redimensionando a imagem 2

max_img1 = np.max(img1)# Valor máximo da imagem 1
max_img2 = np.max(img2)# Valor máximo da imagem 2

for i in range(0, 200):
    img1[i] = max_img1 * img1[i]
    img2[i] = max_img2 * img2[i]
    
img_sum_max = img1 + img2
np.where(img_sum_max > 255, 255, img_sum_max)

img_sum_max = img_sum_max/255# Normalizando

img_sum_max_result = cv2.imshow('Imagens somadas', img_sum_max)
cv2.waitKey(0)
#print(max_img1)
#print(max_img2)

#%%
# Diferenciação absoluta

import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('/home/alunoic/Documentos/lena.png', cv2.IMREAD_COLOR)# Carregando a imagem 1
img1 = cv2.resize(img1, (200,200))# Redimensionando a imagem

img2 = cv2.imread('/home/alunoic/Documentos/baboon.png', cv2.IMREAD_COLOR)# Carregando a imagem 2
img2 = cv2.resize(img2, (200,200))# Redimensionando a imagem 2

img_diff_abs = img1 - img2

np.where(img_diff_abs > 255, 255, img_diff_abs)

img_diff_abs= img_diff_abs/255# Normalizando

img_diff_result = cv2.imshow('Diff absoluta', img_diff_abs)
cv2.waitKey(0)


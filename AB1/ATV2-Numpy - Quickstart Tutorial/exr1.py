#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2


w = 400 #width - largura
h = 400 #height - altura
black_image = np.zeros((h, w, 3)) #Gera a matriz
cv2.imshow('Imagem preta 400 x 400', black_image) #Mostra a imagem
cv2.waitKey(0) #Mostra a imagem

#%%
import numpy as np
import cv2


w = 400 #width - largura
h = 400 #height - altura
gray_image = np.ones((h, w, 3))*128 #Gera a matriz
cv2.imshow('Imagem cinza 400 x 400', gray_image) #imagem
cv2.waitKey(0) #Mostra a imagem



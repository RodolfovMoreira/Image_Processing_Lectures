#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

#Peganndo a imagem
peacy_img = cv2.imread('/home/rodol/Documentos/UFAL/PDI/ATV2-Numpy - Quickstart Tutorial/baboon.png', cv2.IMREAD_COLOR)
#cv2.imshow('Baboon completo', peacy_img)
cv2.imshow('Parte do baboon.png', peacy_img[255:448, 127:320])
cv2.waitKey(0)
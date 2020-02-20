# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:38:33 2018

@author: alunoic
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img_lena = cv2.imread('/home/horacio/Imagens/lena.png', cv2.IMREAD_COLOR)
cv2.imshow('img', img_lena)
cv2.waitKey(0)
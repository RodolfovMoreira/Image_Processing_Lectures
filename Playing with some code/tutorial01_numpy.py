
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cria uma matriz 400x400 do tipo inteiro em cada elemento é 1
img = np.ones((400,400), dtype=int)
img = img*255 # Multiplica cada elemento da matriz por 255 (preto = 255)
img = cv2.imshow('img', img)
print(np.__version__)
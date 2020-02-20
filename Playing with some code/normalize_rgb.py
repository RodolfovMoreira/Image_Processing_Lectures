
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import fetch_mldata
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from numpy import asarray

import matplotlib.pyplot as plt
import numpy as np
import glob as gl
import cv2 as cv
import os


path_db = "/home/rodol/Documentos/4-PROCESSAMENTO_DE_IMAGENS-8-/KeratoDetect/DataBase/Eyes"
# Separar em dois grupos: Treino e Teste

# Carregando imagens
fold_eye = gl.glob(path_db)

imgs_eyes_list = []

for fold in fold_eye:
    for f in gl.glob(fold + '/*.png'):
        imgs_eyes_list.append(f)

read_fold = [] # Arquivo com dados originais das imagens em RGB
read_fold_norm = [] # Arquivo com dados normalizados das imagens em RBG

for img in imgs_eyes_list:
    read_fold.append(cv.imread(img, cv.IMREAD_COLOR))
    read_fold_norm.append(cv.imread(img, cv.IMREAD_COLOR))


for i in range(0,40):
    read_fold_norm[i] = read_fold_norm[i].astype('float32')
    read_fold_norm[i] /= 255.0

# dividir o dataset entre train (75%) e test (25%)
(trainX, testX, trainY, testY) = train_test_split(read_fold_norm,read_fold)
 
# converter labels de inteiros para vetores
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)




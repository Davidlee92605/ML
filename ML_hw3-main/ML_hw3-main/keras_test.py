import numpy as np
import pandas as pd
import torch
import keras
import csv
import tensorflow as tf
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import cv2
import os
import sys
import gdown

url = "https://drive.google.com/u/0/uc?id=1eRNCKGgpZmFF-Eb9Q2o9JiMwdUqwX6c-&export=download"
gdown.download(url)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#path = os.path.join(sys.argv[1] , 'test')

def load():
    test_x = []
    for index in range(7178):
        img = cv2.imread(os.path.join( sys.argv[1] , '{:0>5d}.jpg'.format(index)) ,cv2.IMREAD_GRAYSCALE)
        test_x.append(img)
    test_x = np.array(test_x)
    test_x = test_x.reshape(-1 , 48 , 48 , 1).astype('float32')/255
    

    model = keras.models.load_model('keras_model.h5')
    
    return (test_x , model)


def save(y_predict):
    with open(sys.argv[2] , 'w',newline='')as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id' , 'label'])
        for index , value in enumerate(y_predict):
            writer.writerow([str(index) , value])


if (__name__ == '__main__'):
    (test_x , model) = load()
    y_predict = model.predict(test_x)
    y_predict = np.argmax(y_predict , axis= 1)
    save(y_predict)

    
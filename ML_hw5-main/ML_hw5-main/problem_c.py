import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import cluster
from tensorflow import keras
import random
import torch
import matplotlib.pyplot as plt
import numpy as np

img_list = np.load('trainX.npy')
img_list = img_list.astype('float')/255
y_test = np.load('trainY.npy')
y_test = y_test.astype('float')/255
indexes = [1,2,3,6,7,9, 10,11,12,13,14,15,16,17,18]
img_list = img_list[indexes, ]
y_test = y_test[indexes, ]
#print(img_list.shape)

encoder_model = keras.models.load_model('encoder_model.h5')
print(encoder_model.summary())

encoded_images = encoder_model.predict(img_list)
print(img_list.shape)
print(y_test.shape)

#x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(encoded_images[:, 0], encoded_images[:, 1])
plt.colorbar()
plt.show()
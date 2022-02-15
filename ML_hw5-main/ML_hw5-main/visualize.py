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
indexes = [1,2,3,6,7,9]
img_list = img_list[indexes, ]
print(img_list.shape)

autoencoder_model = keras.models.load_model('autoencoder_model.h5')
print(autoencoder_model.summary())

encoded_images = autoencoder_model.predict(img_list)
print(img_list.shape)
print(encoded_images.shape)


plt.figure(figsize=(10,4))
# indexes = [1,2,3,6,7,9]
# imgs = trainX[indexes,]
for i, img in enumerate(img_list):
    plt.subplot(2, 6, i+1, xticks=[], yticks=[])
    plt.imshow(img)

# 畫出 reconstruct 的圖
# inp = torch.Tensor(trainX_preprocessed[indexes,])
# latents, recs = model(inp)
# recs = ((recs+1)/2 ).cpu().detach().numpy()
# recs = recs.transpose(0, 2, 3, 1)

for i, img in enumerate(encoded_images):
    plt.subplot(2, 6, 6+i+1, xticks=[], yticks=[])
    plt.imshow(img)
  
plt.tight_layout()
plt.show()


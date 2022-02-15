import sys
import numpy as np
import pandas as pd

#from keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import cluster
from tensorflow import keras

def load():
  img_list = np.load('trainX.npy')
  img_list = img_list.astype('float')/255
  return img_list

def cluster():
  pass


if(__name__ == '__main__'):
  img_list = load()
  encoder_model = keras.models.load_model('encoder_model.h5')
  print(encoder_model.summary())
  encoded_images = encoder_model.predict(img_list)
  print('Shape Before PCA: ',encoded_images.shape)

  sample , x , y , z = encoded_images.shape
  train_img = encoded_images.reshape((sample , x * y * z))
  
  #train_img = (train_img - np.mean(train_img , axis = 0)) / np.std(train_img , axis = 0)

  pca = PCA(n_components=100 , whiten=True )
  PCA_data=pca.fit_transform(train_img)

  tsne = TSNE(n_components = 2)
  tsne_data = tsne.fit_transform(PCA_data)

  K_result = KMeans(n_clusters=2 ,max_iter=800, n_init=10, verbose=1, n_jobs=-1).fit(tsne_data)
  ans = np.array(K_result.labels_)

  if (np.sum(ans[ : 5]) > 2):
	  ans = 1 - ans
  
  number_of_data = ans.shape[0]
  df = pd.DataFrame({'id' : np.arange(number_of_data) , 'label' : ans})
  df.to_csv('ans.csv' , index = False)


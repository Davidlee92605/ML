import torch 
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam , Adadelta
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pickle
def load_testing_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
       
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
        print(X[:5])
    return X

class DNN_0(nn.Module):
	def __init__(self , dimension):
		super(DNN_0 , self).__init__()

		self.linear = nn.Sequential(
			nn.Linear(dimension , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.ReLU() ,
			nn.Dropout() ,
			nn.Linear(100 , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.ReLU() ,
			nn.Dropout() ,
			nn.Linear(100 , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.ReLU() ,
			nn.Dropout() ,
			nn.Linear(100 , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.ReLU() ,
			nn.Dropout() ,
			nn.Linear(100 , 1 , bias = True),
			nn.Sigmoid()
		)

		return

	def forward(self , x):
		x = self.linear(x)
		return x

def word_dictionary(X_test):
	with open('tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)

	vectors = tokenizer.texts_to_matrix(X_test, mode='count')
	print("Vocabulary size: ", vectors[0].shape)
	
	return vectors

def test(test_x , model):
	batch_size = 1024
	threshold = 0.5

	test_x = torch.FloatTensor(test_x)
	print(test_x.shape)
	test_loader = DataLoader(test_x , batch_size = batch_size , shuffle = False)

	model.eval()
	predict = list()
	with torch.no_grad():
		for data in test_loader:
			#data = data.to(device)
			y = model(data)
			result = (y > threshold).int()*1
			predict.append(result.detach().numpy())
	test_y = np.concatenate(predict , axis = 0)

	return test_y

def dump(y):
	number_of_data = y.shape[0]
	y = np.array(y)
	y = y.ravel()
	df = pd.DataFrame({'id' : np.arange(number_of_data) , 'label' : y})
	df.to_csv('predict_DNN.csv' , index = False)
	return


if (__name__ == '__main__'):
	path_test ='testing_data.txt'
	X_test = load_testing_data(path_test)
	X_test = word_dictionary(X_test)
	model = DNN_0(X_test.shape[1])
	model = torch.load('bow_dnn.pt')
	print(model)
	count = 0
	for parameter in model.parameters():
		print(parameter.shape)
	Y_test = test(X_test , model)
	dump(Y_test)

    
	
	


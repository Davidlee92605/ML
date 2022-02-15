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
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class dataset(Dataset):
	def __init__(self , data , label):
		self.data = data
		self.label = label
		return

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self , index):
		return (torch.FloatTensor(self.data[index]) , self.label[index])
################################### load data ###################################
def load_labled_training_data(path):
    print("Start loading training data...")
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip('\n').split(' ') for line in lines]
    x_train = [line[2:] for line in lines]
    y_train = [int(line[0]) for line in lines]
    return x_train, y_train

def load_testing_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
       
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
        print(X[:5])
    return X
################################# construct BOW #################################
def word_dictionary(X_train):
	print("Start constructing vocabulary...")
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(X_train)

	vectors = tokenizer.texts_to_matrix(X_train, mode='count')
	print("Vocabulary size: ", vectors[0].shape)
	
	with open('tokenizer.pickle', 'wb') as handle:
		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	return vectors

class DNN_0(nn.Module):
	def __init__(self , dimension):
		super(DNN_0 , self).__init__()

		self.linear = nn.Sequential(
			nn.Linear(dimension , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.Sigmoid() ,
			nn.Dropout() ,
			nn.Linear(100 , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.Sigmoid() ,
			nn.Dropout() ,
			nn.Linear(100 , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.Sigmoid() ,
			nn.Dropout() ,
			nn.Linear(100 , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.Sigmoid() ,
			nn.Dropout() ,
			nn.Linear(100 , 1 , bias = True),
			nn.Sigmoid()
		)

		return

	def forward(self , x):
		x = self.linear(x)
		return x


def train(X_train , Y_train ,X_val , Y_val , model):
	learning_rate = 0.0001
	epoch = 50
	batch_size = 1024
	loss_values = []
	val_loss_values = []
	threshold = torch.tensor([0.5])

	train_dataset = dataset(X_train , Y_train)
	val_dataset = dataset(X_val , Y_val)
	train_loader = DataLoader(train_dataset , batch_size = 1024 , shuffle = True)
	validation_loader = DataLoader(val_dataset , batch_size = 1024 , shuffle = False)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate ) 


	for i in range(epoch):
		model.train()
		total_loss = 0
		val_total_loss = 0

		for (j , (data , label)) in enumerate(train_loader):
			optimizer.zero_grad()
			y_predicted = model(data)
			label = np.asarray(label) 
			label = torch.from_numpy(label)
			y_predicted = y_predicted.to(torch.float32)
			label = label.to(torch.float32)
			loss = F.binary_cross_entropy(y_predicted , label)
			total_loss += loss.item()

			# Backward pass and update
			loss.backward()
			optimizer.step()

			# zero grad before new step
			optimizer.zero_grad()

			if (j < len(train_loader) - 1):
				n = (j + 1) * batch_size
				m = int(50 * n / X_train.shape[0])
				bar = m * '=' + '>' + (49 - m) * ' '
				print('epoch {}/{} ({}/{}) [{}]'.format(i + 1 , epoch , n ,X_train.shape[0] , bar) , end = '\r')
			else:
				n = X_train.shape[0]
				bar = 50 * '='
				print('epoch {}/{} ({}/{}) [{}]  loss : {:.8f}'.format(i + 1 , epoch , n , X_train.shape[0] , bar  , total_loss / X_train.shape[0]))
			
		loss_values.append(total_loss / X_train.shape[0])
		
		model.eval()
		with torch.no_grad():
			for (j , (data , label)) in enumerate(validation_loader):
				y_predicted = model(data)
				label = np.asarray(label) 
				label = torch.from_numpy(label)
				y_predicted = y_predicted.to(torch.float32)
				label = label.to(torch.float32)
				loss = F.binary_cross_entropy(y_predicted , label)
				val_total_loss += loss.item()

		val_loss_values.append(val_total_loss / X_val.shape[0])


		if ((i + 1) % 10 == 0):
			model.eval()
			predict = list()
			with torch.no_grad():
				for (data , label) in validation_loader:
					y = model(data)
					result = (y > threshold).float()*1
					print(result)
					predict.append(result.detach().numpy())
			predict = np.concatenate(predict , axis = 0)
			validation_score = f1_score(Y_val , predict , average = 'micro')
			print('evaluation validation F1-score : {:.5f}'.format(validation_score))
	
	return model , loss_values , val_loss_values
    

if (__name__ == '__main__'):
	path = 'training_label.txt'
	path_test ='testing_data.txt'
	(X_train, Y_train) = load_labled_training_data(path)
	X_test = load_testing_data(path_test)

	print(X_train[0])
	print(X_test[0])
	X_train = word_dictionary(X_train[:7000])
	Y_train = Y_train[:7000]
	
	X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.15 , random_state = 1,shuffle = True)

	model = DNN_0(X_train.shape[1])
	(model , loss_value , val_loss_value) = train(X_train , Y_train , X_val , Y_val , model)
	torch.save(model,'bow_dnn.pt')
	plt.title("Loss Curve") 
	plt.xlabel("epoch") 
	plt.ylabel("loss")
	plt.plot(loss_value , '-b' , label= 'Training_loss')
	plt.plot(val_loss_value , '-r' , label= 'Validation_loss')
	plt.legend(loc='upper right')
	plt.show()

	


    #print(len(Y_train))
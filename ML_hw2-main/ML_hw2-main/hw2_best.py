import sys
import numpy as np 
import csv
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib

########################## load data ###############################
df_x_0 = pd.read_csv(sys.argv[1])
data_x_0 = df_x_0[["education_num"]]
df_x = pd.read_csv(sys.argv[2])
#data_x = df_x.drop(["fnlwgt"] , axis=1)
data_x = df_x
temp = pd.concat([data_x_0 , data_x] , axis=1 )
data_x = temp.values

with open(sys.argv[3] , 'r') as file:
	data_y = np.array([line for line in file]).astype(np.int)

train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, train_size=.95)

########################## normalize ###############################

mean = train_x.mean(axis = 0)
std = train_x.std(axis = 0)
train_x = (train_x - mean) / std

val_x = (val_x - mean) / std

######################## train ###############################
#validation_fraction = 0.2, n_iter_no_change = 20 , tol = 0.00001 , min_samples_split=500,min_samples_leaf=70,max_depth=8
model = GradientBoostingClassifier(loss = 'deviance',
learning_rate = 0.15 , n_estimators = 500 , min_samples_split=900, min_samples_leaf=60, max_depth=3)
model.fit(train_x ,train_y)

print('train accuracy : {:.5f}'.format(model.score(train_x , train_y)))
print('validation accuracy : {:.5f}'.format(model.score(val_x , val_y)))
print(model.classes_)
######################## output ###############################
np.save('best_mean.npy' , mean)
np.save('best_std.npy' , std)
joblib.dump(model , 'best_model.joblib')
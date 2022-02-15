import sys
import numpy as np 
import csv
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
import joblib

########################## load data ###############################

mean = np.load('best_mean.npy')
std = np.load('best_std.npy')
model = joblib.load('best_model.joblib')

########################## load data ###############################
df_x_0 = pd.read_csv(sys.argv[1])
data_x_0 = df_x_0[["education_num"]]
df_test = pd.read_csv(sys.argv[2])
#data_x = df_test.drop(["fnlwgt"] , axis=1)
data_x = df_test
temp = pd.concat([data_x_0 , data_x] , axis=1 )
data_x = temp.values
train_x = data_x


########################## normalize ###############################
train_x = (train_x - mean) / std


########################## predict ###############################
id_value = model.predict(train_x)



#########################   output   #########################

with open(sys.argv[3] , 'w' ,newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id' , 'label'])
    for index , value in enumerate(id_value):
        writer.writerow([str(index + 1) , value])

########################## load compare ###############################
with open('compare.csv' , 'r') as file:
	data_y = np.array([line for line in file]).astype(np.int)
count = 0
for index , value in enumerate(id_value):
    if(id_value[index] == data_y[index]):
        count += 1
print(count)
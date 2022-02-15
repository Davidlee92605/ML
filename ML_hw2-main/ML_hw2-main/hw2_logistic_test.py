import sys
import numpy as np 
import csv
import pandas as pd

########################## load data ###############################

mean = np.load('logistic_mean.npy')
std = np.load('logistic_std.npy')
model = np.load('logistic_model.npy')
weight = model[ : -1]
bias = model[-1]

########################## load data ###############################
df_x_0 = pd.read_csv(sys.argv[2])
data_x_0 = df_x_0[["education_num"]]
df_test = pd.read_csv(sys.argv[5])
data_x = df_test.drop(["fnlwgt" ,  "hours_per_week" ,"sex"] , axis=1)
data_x = df_test
temp = pd.concat([data_x_0 , data_x] , axis=1 )
data_x = temp.values
train_x = data_x



########################## normalize ###############################
train_x = (train_x - mean) / std
# def _normalize_column_normal(X, train=True, specified_column = None, X_mean=None, X_std=None):
#     # The output of the function will make the specified column number to 
#     # become a Normal distribution
#     # When processing testing data, we need to normalize by the value 
#     # we used for processing training, so we must save the mean value and 
#     # the variance of the training data
#     if train:
#         if specified_column == None:
#             specified_column = np.arange(X.shape[1])
#         length = len(specified_column)
#         X_mean = np.reshape(np.mean(X[:, specified_column],0), (1, length))
#         X_std  = np.reshape(np.std(X[:, specified_column], 0), (1, length))
    
#     X[:,specified_column] = np.divide(np.subtract(X[:,specified_column],X_mean), X_std)

# col = [0,1,2,4,5,6,8,11,13,26,27,28,29 , 30 ,46,48,73,82]
# _normalize_column_normal(train_x,train=False, specified_column=col , X_mean=mean, X_std=std)
########################## sigmoid ###############################
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

########################## accuracy ###############################

def accuracy(x , y , weight , bias):
    count = 0
    number_of_data = x.shape[0]
    for i in range(number_of_data):
        probability = sigmoid(weight @ x[i] + bias)
        if ((probability > 0.5 and y[i] == 1) or (probability < 0.5 and y[i] == 0)):
            count += 1
    return count * 100 / number_of_data

########################## predict ###############################
id_value = []
for i in range(train_x.shape[0]):
    if(sigmoid(weight @ train_x[i] + bias) > 0.5):
        id_value.append(1)
    else:
        id_value.append(0)

#########################   output   #########################

with open(sys.argv[6] , 'w' ,newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id' , 'label'])
    for index , value in enumerate(id_value):
        writer.writerow([str(index + 1) , value])
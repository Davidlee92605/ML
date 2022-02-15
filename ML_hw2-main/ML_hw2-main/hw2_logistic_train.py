import sys
import numpy as np 
import csv
import pandas as pd
#" Self-emp-inc"," Masters"," Prof-school"," Doctorate"," Exec-managerial"," Husband"," Wife"," Prof-specialty"," White" , " Federal-gov" , " Asian-Pac-Islander"
#"age","fnlwgt"," Self-emp-inc"," Masters"," Prof-school"," Doctorate"," Exec-managerial"," Husband"," Wife"," Prof-specialty"," White" , " Federal-gov" , " Asian-Pac-Islander" ,"capital_gain", "capital_loss", "hours_per_week"
# Divorced, Married-AF-spouse, Married-civ-spouse, Married-spouse-absent
# ," Self-emp-inc"," Masters"," Prof-school"," Doctorate"," Exec-managerial"," Husband"," Wife"," Prof-specialty"," White" , " Asian-Pac-Islander", " Married-civ-spouse"  , " Protective-serv"," Preschool" , " Without-pay" , " 1st-4th"," 5th-6th"," 7th-8th"," 9th" , " Never-married"
########################## load data ###############################
df_x_0 = pd.read_csv(sys.argv[1])
data_x_0 = df_x_0[["education_num"]]
df_x = pd.read_csv(sys.argv[2])
data_x = df_x.drop(["fnlwgt" , "hours_per_week" , "sex"] , axis=1)
data_x = df_x
temp = pd.concat([data_x_0 , data_x] , axis=1 )
data_x = temp.values
with open(sys.argv[3] , 'r') as file:
	data_y = np.array([line for line in file]).astype(np.int)

number = data_x.shape[0]
index = np.arange(number)
np.random.shuffle(index)

train_x = data_x[index]
train_y = data_y[index]

val_x = train_x[:1000]
val_y = train_y[:1000]
train_x = train_x[1000:]
train_y = train_y[1000:]

########################## normalize ###############################
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
     
#     return X, X_mean, X_std

# #no edu_num [0,1,3,4,5,7,10,12,25,26,27,28]
# #edu_num [0,1,2,4,5,6,8,11,13,26,27,28,29]
# col = [0,1,2,4,5,6,8,11,13,26,27,28,29]
# train_x , mean , std = _normalize_column_normal(train_x, specified_column=col)
# val_x , mean ,std = _normalize_column_normal(val_x,train=False, specified_column=col , X_mean=mean, X_std=std)
mean = train_x.mean(axis = 0)
std = train_x.std(axis = 0)
train_x = (train_x - mean) / std

val_x = (val_x - mean) / std

######################## sigmoid ###############################
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

######################## accuracy ###############################

def accuracy(x , y , weight , bias):
    count = 0
    number_of_data = x.shape[0]
    for i in range(number_of_data):
        probability = sigmoid(weight @ x[i] + bias)
        if ((probability > 0.5 and y[i] == 1) or (probability < 0.5 and y[i] == 0)):
            count += 1
    return count * 100 / number_of_data


######################## train ###############################
n_samples , n_features = train_x.shape

n_iter = 5
weights = np.zeros(n_features)
bias = 0.5


lr = 0.8
w_lr = np.ones(train_x.shape[1])
b_lr = 0

for i in range(n_iter):
    number = train_x.shape[0]
    index = np.arange(number)
    np.random.shuffle(index)
    train_x = train_x[index]
    train_y = train_y[index]

    linear_model = np.dot(train_x ,weights) + bias #sample * 4
    y_predict = sigmoid(linear_model)

    dw =  np.dot(train_x.T , (y_predict - train_y)) # 4 * sample , sample * 1 = 4 * 1
    db =  np.sum(y_predict - train_y)
    #(1 / n_samples) * 
    w_lr += dw**2
    b_lr += db**2

    weights -= lr / np.sqrt(w_lr + 1e-100) * dw
    bias -= lr/np.sqrt(b_lr + 1e-100) * db

    loss = -1 * np.mean(train_y * np.log(y_predict + 1e-100) + (1 - train_y) * np.log(1 - y_predict + 1e-100))
    train_accuracy = accuracy(train_x , train_y , weights , bias)
    val_train_accuracy = accuracy(val_x , val_y , weights , bias)
    print("loss : {loss}" .format(loss = loss))
    print("Accuracy : {train_accuracy}" .format(train_accuracy = train_accuracy))
    print("val_Accuracy : {val_train_accuracy}" .format(val_train_accuracy = val_train_accuracy))

######################## output ###############################
model = np.hstack((weights , bias))
np.save('logistic_mean.npy' , mean)
np.save('logistic_std.npy' , std)
np.save('logistic_model.npy' , model)
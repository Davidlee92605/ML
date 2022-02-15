import sys
import numpy as np 
import csv
import pandas as pd

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

mean = train_x.mean(axis = 0)
std = train_x.std(axis = 0)
train_x = (train_x - mean) / std

val_x = (val_x - mean) / std


######################## find mean & covariance ###############################
 
class_1  =   np.array([train_x[index] for index in range(train_x.shape[0]) if train_y[index] == 0 ])
class_2  =   np.array([train_x[index] for index in range(train_x.shape[0]) if train_y[index] == 1 ])

num_of_data_1 = class_1.shape[0]
num_of_data_2 = class_2.shape[0]
mean_1 = class_1.mean(axis= 0)
mean_2 = class_2.mean(axis= 0)

ans_1 = np.zeros((class_1.shape[1] , class_1.shape[1]))
for i in range(num_of_data_1):
    temp0 = (class_1[i , :] - mean_1).reshape(-1 , 1) # 107 * 1
    temp1 = temp0.reshape(1 , -1) # 1 * 107
    ans_1 += temp0.dot(temp1)
cov_1 = ans_1 / num_of_data_1

ans_2 = np.zeros((class_2.shape[1] , class_2.shape[1]))
for i in range(num_of_data_2):
    temp0 = (class_2[i , :] - mean_2).reshape(-1 , 1) # 107 * 1
    temp1 = temp0.reshape(1 , -1) # 1 * 107
    ans_2 += temp0.dot(temp1)
    
cov_2 = ans_2 / num_of_data_2

cov = (num_of_data_1 / (num_of_data_1 + num_of_data_2)) * cov_1 + (num_of_data_2 / (num_of_data_1 + num_of_data_2)) * cov_2
######################## Gaussian ###############################
def Gaussian(data_x , mean , inv_cov):
    ans = np.exp(-0.5 * (data_x - mean).reshape(1 , -1) @ inv_cov @ (data_x - mean).reshape(-1 , 1))
    return ans

prior_probability_1 = (num_of_data_1 / (num_of_data_1 + num_of_data_2))
prior_probability_2 = (num_of_data_2 / (num_of_data_1 + num_of_data_2))
######################## accuracy ###############################
def accuracy(prior_probability_1 , prior_probability_2 , cov , train_x , train_y , mean_1 , mean_2):
    inv_cov = np.linalg.inv(cov)
    count = 0
    for i in range(train_x.shape[0]):
        probability_1 = prior_probability_1 * Gaussian(train_x[i , :] , mean_1 , inv_cov)
        probability_2 = prior_probability_2 * Gaussian(train_x[i , :] , mean_2 , inv_cov)
        if ((probability_1 > probability_2 and train_y[i] == 0) or (probability_1 < probability_2 and train_y[i] == 1)):
                count += 1
    ans = count * 100 / train_x.shape[0]
    return ans

print(accuracy(prior_probability_1 , prior_probability_2 , cov , train_x , train_y , mean_1 , mean_2))
print(accuracy(prior_probability_1 , prior_probability_2 , cov , val_x , val_y , mean_1 , mean_2))
######################## output ###############################
np.save('generative_mean.npy' , mean)
np.save('generative_std.npy' , std)
np.save('generative_prob_1.npy' , prior_probability_1)
np.save('generative_prob_2.npy' , prior_probability_2)
np.save('generative_mean_1.npy' , mean_1)
np.save('generative_mean_2.npy' , mean_2)
np.save('generative_cov.npy' , cov)
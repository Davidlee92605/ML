import sys
import numpy as np
import csv
import pandas as pd

########################## load model ###############################
mean = np.load('generative_mean.npy')
std = np.load('generative_std.npy')

prior_prob_1 = np.load('generative_prob_1.npy')
prior_prob_2 = np.load('generative_prob_2.npy')
mean_1 = np.load('generative_mean_1.npy')
mean_2 = np.load('generative_mean_2.npy')
cov = np.load('generative_cov.npy')

########################## load data ###############################
df_x_0 = pd.read_csv(sys.argv[2])
data_x_0 = df_x_0[["education_num"]]
df_test = pd.read_csv(sys.argv[5])
#data_x = df_test.drop(["fnlwgt" ,  "hours_per_week" ,"sex"] , axis=1)
data_x = df_test
temp = pd.concat([data_x_0 , data_x] , axis=1 )
data_x = temp.values
test_x = data_x

########################## normalize ###############################
test_x = (test_x - mean) / std

########################## Gaussian ################################
def gaussian(x , mean , covariance):
    inv_cov = np.linalg.inv(covariance)
    return np.exp(-0.5 * ((x - mean).T @ inv_cov) @ (x - mean))

########################### predict ################################
id_value = []
for i in range(test_x.shape[0]):
    prob_1 = prior_prob_1 * gaussian(test_x[i].reshape((-1 , 1)), mean_1.reshape(-1, 1), cov)
    prob_2 = prior_prob_2 * gaussian(test_x[i].reshape((-1 , 1)), mean_2.reshape(-1, 1), cov)
    if(prob_1 > prob_2):
        id_value.append(0)
    else:
        id_value.append(1)

############################# output ###############################
with open(sys.argv[6] , 'w' ,newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id' , 'label'])
    for index , value in enumerate(id_value):
        writer.writerow([str(index + 1) , value])
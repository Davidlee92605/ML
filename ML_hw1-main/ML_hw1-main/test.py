import sys
import csv
import numpy as np
import torch
import torch.nn as nn

######################### load data #########################
data1 = open(sys.argv[1] , 'r' , encoding = 'unicode_escape')
test_data = csv.reader(data1 , delimiter = ',')

storage = []
clean_data = []
cnt_1 = 0

for row in test_data:
    if(cnt_1 != 0 ):
        append_flag = True
        zero = False
        for i in range(len(row)):
            if(i in [0 , 2 , 3 , 6 , 9 , 10]):      
                try:
                    row[i] = float(row[i])
                except ValueError:
                    append_flag = False
                    break
        if(append_flag == True):
            storage.append(row)
    cnt_1 += 1
    

for index , row in enumerate(storage):
    if(cnt_1 != 0 ):
        append_flag = True
        zero = False
        for i in range(len(row)):
            if(i in [0 , 2 , 3 , 6 , 9 , 10]):      
                try:
                    row[i] = float(row[i])
                    assert row[i] != 0
                except ValueError:
                    append_flag = False
                    break
                except AssertionError:
                    zero = True
                    append_flag = True
                    break
        if(append_flag == True):
            temp = []
            if(zero == True):
                if(index == len(storage)-1 ):
                    for i in range(len(row)):
                        if(i in [0 , 2 , 3 , 6 , 9 , 10]):
                            temp.append(storage[index - 1][i])
                elif(index == 1):
                    for i in range(len(row)):
                        if(i in [0 , 2 , 3 , 6 , 9 , 10]):
                            temp.append(storage[index + 1][i])
                else:
                    for i in range(len(row)):
                        if(i in [0 , 2 , 3 , 6 , 9 , 10]):
                            temp.append((storage[index - 1][i] + storage[index + 1][i]) / 2)
                clean_data.append(temp)  
            else:
                for i , value in enumerate(row):
                    if(i == 0 and value > 4.5 and index != 1 and index != len(storage) - 1):
                        row[i] = (storage[index - 1][i] + storage[index + 1][i]) / 2
                    if(i == 2 and value > 100 and index != 1 and index != len(storage) - 1):
                        row[i] = (storage[index - 1][i] + storage[index + 1][i]) / 2
                    if(i == 3 and value > 55 and index != 1 and index != len(storage) - 1):
                        row[i] = (storage[index - 1][i] + storage[index + 1][i]) / 2
                    if(i == 6 and value > 3.5 and index != 1 and index != len(storage) - 1):
                        row[i] = (storage[index - 1][i] + storage[index + 1][i]) / 2
                    if(i == 9 and value > 110 and index != 1 and index != len(storage) - 1):
                        row[i] = (storage[index - 1][i] + storage[index + 1][i]) / 2
                    if(i == 10 and value > 65 and index != 1 and index != len(storage) - 1):
                        row[i] = (storage[index - 1][i] + storage[index + 1][i]) / 2
                for i in range(len(row)):
                    if(i in [0 , 2 , 3 , 6 , 9 , 10]):
                        temp.append(row[i])
                
                clean_data.append(temp)
    cnt_1 += 1

print(len(clean_data))
######################### load model #########################
c = np.load('model.npy')
weight = []
bias = c[54]

for i in range(9):
    weight.append([])
    for j in range(6):
        weight[i].append(c[i*6 + j])
#########################   predict  #########################
predict = []
X = []
for i in range(500):
    X.append([])
    for j in range(9):
        X[i].append(clean_data[i*9 + j])

id_value = []

for i in range(500):
    ans = 0
    for j in range(9):
        for k in range(6):
            temp = X[i][j][k] * weight[j][k]
            ans += temp
    id_value.append(ans + bias)


#########################   output   #########################

with open(sys.argv[2] , 'w' ,newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id' , 'value'])
    for index , value in enumerate(id_value):
        writer.writerow(['id_' + str(index) , value])

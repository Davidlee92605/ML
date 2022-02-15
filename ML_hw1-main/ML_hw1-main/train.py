import sys
import csv
import numpy as np
import torch
import torch.nn as nn

#read dataset in
raw_data1 = open(sys.argv[1], 'r' , encoding = 'unicode_escape')
raw_data2 = open(sys.argv[2], 'r' , encoding = 'unicode_escape')

rows_1 = csv.reader(raw_data1, delimiter = ',')
rows_2 = csv.reader(raw_data2, delimiter = ',')

clean_data = list()
remove_row = False
cnt_1 = 0
cnt_2 = 0

#remove data with '' or '0' or '-' and get a clean data 
for row in rows_1:
    if cnt_1 != 0:
        append_flag = True
        for i in range(len(row)):
            if(i in [0 , 2 , 3 , 6 , 9 , 10]):        
                try:
                    row[i] = float(row[i])
                    assert row[i] != 0
                except ValueError:
                    append_flag = False
                    break
                except AssertionError:
                    append_flag = False
                    break
                if(row[i] > 500 ):
                    append_flag = False
                    break
            
        if(append_flag == True):
            temp = []
            for i in range(len(row)):
                if(i in [0 , 2 , 3 , 6 , 9 , 10]):
                    temp.append(row[i])
            #clean_data.append(temp)
            if(temp[0] < 4): #0
                if(temp[1] < 80): #3
                    if(temp[2] < 45): #4
                        if(temp[3] < 3.5): #5
                            if(temp[4] < 100): #9
                                if(temp[5] < 65): #10
                                    clean_data.append(temp)
    cnt_1 += 1


for row in rows_2:
    if cnt_2 != 0:
        append_flag = True
        for i in range(len(row)):
            if(i in [0 , 2 , 3 , 6 , 9 , 10]):         
                try:
                    row[i] = float(row[i])
                    assert row[i] != 0
                except ValueError:
                    append_flag = False
                    break
                except AssertionError:
                    append_flag = False
                    break
                if(row[i] > 500 ):
                    append_flag = False
                    break
                if(row[i]  == 0.9865591397849462):
                    append_flag = False
                    break

        if(append_flag == True):
            temp = []
            for i in range(len(row)):
                if(i in [0 , 2 , 3 , 6 , 9 , 10]):
                    temp.append(row[i])
            #clean_data.append(temp)          
            if(temp[0] < 4): #0
                if(temp[1] < 80): #3
                    if(temp[2] < 45): #4
                        if(temp[3] < 3.5): #5
                            if(temp[4] < 100): #9
                                if(temp[5] < 65): #10
                                    clean_data.append(temp)
    cnt_2 += 1

print(len(clean_data))


validation_x = clean_data[ -2000 : -1]
validation_y = clean_data[ -2000 : -1]

######################################################################

# Linear regression
# f = w * x + b
x = []
y = []
for i in range(15000):
    x.append([])
    y.append([])
    for f in range(6):
        for s in range(9):
            x[i].append(clean_data[i+s][f])

    y[i].append(clean_data[i+9][5])

val_x = []
val_y = []
for i in range(1990):
    val_x.append([])
    val_y.append([])
    for j in range(6):
        for k in range(9):
            val_x[i].append(validation_x[i+k][j])
    val_y[i].append(validation_y[i+9][5])

# 1) Model
# Linear model f = wx + b
input_size = 54
output_size = 1
model = nn.Linear(input_size , output_size)

# 2) Loss and optimizer
learning_rate = 0.00004
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate ) 

# # 3) Training loop
num_epochs = 20000
X = torch.tensor(x , dtype=torch.float32)
Y = torch.tensor(y , dtype=torch.float32)

val_X = torch.tensor(val_x , dtype=torch.float32)
val_Y = torch.tensor(val_y , dtype=torch.float32)

for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_predicted = model(X)
        val_y_predicted = model(val_X)
        loss = torch.sqrt(criterion(y_predicted , Y))
        val_loss = torch.sqrt(criterion(val_y_predicted , val_Y))

        # Backward pass and update
        loss.backward()
        val_loss.backward()
        optimizer.step()

        # zero grad before new step
        optimizer.zero_grad()

        #if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
        print(f'val_epoch: {epoch+1}, val_loss = {val_loss.item():.4f}')

a = list(model.parameters())
temp0 = a[0].detach().numpy()
temp0 = temp0[0]
model = np.hstack((temp0 , a[1].detach().numpy()))
np.save('model.npy' , model)

######## validation #########


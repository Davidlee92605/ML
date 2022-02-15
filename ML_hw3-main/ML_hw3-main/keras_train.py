import numpy as np
import pandas as pd
import torch
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten , BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam , Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau , EarlyStopping
import cv2
import os
import sys



os.environ['KMP_DUPLICATE_LIB_OK']='True'
#path = os.path.join(sys.argv[1] , 'train')
#y_path = os.path.join(str(sys.argv[2]) , 'label.csv')

def load():
    train_x = []
    for index in range(28709):
        img = cv2.imread(os.path.join( sys.argv[1] , '{:0>5d}.jpg'.format(index)) ,cv2.IMREAD_GRAYSCALE)
        train_x.append(img)
    train_x = np.array(train_x)
    train_x = train_x.reshape(-1 , 48 , 48 , 1).astype('float')/255
    print(train_x.shape)
    df_y = pd.read_csv(sys.argv[2])
    df_y = df_y.drop(['id'] , axis = 1)
    train_y = df_y.values
    train_y = to_categorical(train_y)

    number = train_x.shape[0]
    index = np.arange(number)
    np.random.shuffle(index)

    train_x = train_x[index]
    train_y = train_y[index]
    val_x = train_x[:1000]
    val_y = train_y[:1000]
    train_x = train_x[1000:]
    train_y = train_y[1000:]



    return (train_x , train_y , val_x , val_y)

def train(train_x , train_y , val_x , val_y):
    train_datagen = ImageDataGenerator(featurewise_center=False,
                                featurewise_std_normalization=False,
                                rotation_range=40,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                zoom_range=0.1,
                                horizontal_flip = True,
                                fill_mode='nearest'
                              )
    train_datagen.fit(train_x)

    LR_function = ReduceLROnPlateau(monitor='val_acc',
                                patience= 2,
                                verbose= 1,
                                factor= 0.9
                                )

    model = Sequential()
    #first conv
    model.add(Conv2D(32 , (3, 3) , activation='relu',padding= 'same' ,input_shape= (48 , 48 ,1)  ))
    model.add(BatchNormalization())
    model.add(Conv2D(32 , (3, 3) , activation='relu',padding= 'same' ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2) , strides=(2,2) ))
    model.add(Dropout(0.5))


    #3rd conv
    model.add(Conv2D(64 , (3, 3) , activation='relu',padding= 'same' ))
    model.add(BatchNormalization())
    model.add(Conv2D(64 , (3, 3) , activation='relu',padding= 'same' ))
    model.add(BatchNormalization())
    model.add(Conv2D(64 , (3, 3) , activation='relu',padding= 'same' ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2) ))
    model.add(Dropout(0.5))


    #5th conv
    model.add(Conv2D(128 , (3, 3) , activation='relu',padding= 'same' ))
    model.add(BatchNormalization())
    model.add(Conv2D(128 , (3, 3) , activation='relu',padding= 'same' ))
    model.add(BatchNormalization())
    model.add(Conv2D(128 , (3, 3) , activation='relu',padding= 'same' ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2) ))
    model.add(Dropout(0.5))


    #7th conv
    model.add(Conv2D(256 , (3, 3) , activation='relu',padding= 'same' ))
    model.add(BatchNormalization())
    model.add(Conv2D(256 , (3, 3) , activation='relu',padding= 'same' ))
    model.add(BatchNormalization())
    model.add(Conv2D(256 , (3, 3) , activation='relu',padding= 'same' ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2) ))
    model.add(Dropout(0.5))

    model.add(Conv2D(512 , (3, 3) , activation='relu',padding= 'same' ))
    model.add(BatchNormalization())
    model.add(Conv2D(512 , (3, 3) , activation='relu',padding= 'same' ))
    model.add(BatchNormalization())
    model.add(Conv2D(512 , (3, 3) , activation='relu',padding= 'same' ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2) ))
    model.add(Dropout(0.5))


    model.add(Flatten())

    #Fully connected
    model.add(Dense(units = 512 , activation= 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(units = 256 , activation= 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(units = 128 , activation= 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(units = 64 , activation= 'relu'))
    model.add(BatchNormalization())

    model.add(Dense(units = 7 , activation = 'softmax'))



    model.compile(optimizer =Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    model.summary()
    # history = model.fit_generator(train_datagen.flow(train_x , train_y, batch_size = 64) , 
    #                             steps_per_epoch=train_x.shape[0] / 64 , epochs= 100, validation_data=(val_x , val_y),
    #                             callbacks= [LR_function])

    # result_train = model.evaluate(train_x,train_y)
    # print('Train Acc:', result_train[1])
    # result_val = model.evaluate(val_x,val_y)
    # print('Val Acc:', result_val[1])

    # hist_df = pd.DataFrame(history.history)
    # model.save('keras_model.h5')
    

if (__name__ == '__main__'):
    (train_x , train_y , val_x , val_y) = load()
    train(train_x ,train_y , val_x , val_y )

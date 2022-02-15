import sys
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.optimizers import RMSprop , Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization 
from tensorflow.keras.layers import Conv2DTranspose, Activation, Reshape, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K

def load():
  img_list = np.load('trainX.npy')
  img_list = img_list.astype('float')/255
  return img_list


def model(img_list):

    LR_function = ReduceLROnPlateau(monitor='loss',
                                patience= 2,
                                verbose= 1,
                                factor= 0.9
                                )

    model = Sequential()
    #Encoder:
    input_img= Input(shape= (32 , 32 ,3))
    conv_1 = Conv2D(64 , (3, 3) , activation='relu',padding= 'same')(input_img)
    bn_1 = BatchNormalization()(conv_1)
    mp_1 = MaxPooling2D(pool_size=(2,2)  , strides=(2,2))(bn_1)
    conv_6 = Conv2D(128, (3, 3) , activation='relu',padding= 'same')(mp_1)
    bn_6 = BatchNormalization()(conv_6)
    mp_6 = MaxPooling2D(pool_size=(2,2)  , strides=(2,2))(bn_6)
    conv_2 = Conv2D(256 , (3, 3) , activation='relu',padding= 'same')(mp_6)
    bn_2 = BatchNormalization()(conv_2)
    mp_2 = MaxPooling2D(pool_size=(2,2)  , strides=(2,2))(bn_2)
    # conv_7 = Conv2D(256, (3, 3) , activation='relu',padding= 'same')(mp_2)
    # bn_7 = BatchNormalization()(conv_7)
    # mp_7 = MaxPooling2D(pool_size=(2,2)  , strides=(2,2))(bn_7)
    
    #Decoder:
    
    us_3 = UpSampling2D((2,2) )(mp_2)
    conv_3 = Conv2DTranspose(256 , (3,3) ,activation='relu' ,padding= 'same' )(us_3)
    bn_3 = BatchNormalization()(conv_3)
    # us_8 = UpSampling2D((2,2) )(bn_3)
    # conv_8 = Conv2DTranspose(256 , (3,3) ,activation='relu' ,padding= 'same' )(us_8)
    # bn_8 = BatchNormalization()(conv_8)
    us_9 = UpSampling2D((2,2) )(bn_3)
    conv_9 = Conv2DTranspose(128 , (3,3) ,activation='relu' ,padding= 'same' )(us_9)
    bn_9 = BatchNormalization()(conv_9)
    us_4 = UpSampling2D((2,2) )(bn_9)
    conv_4 = Conv2DTranspose(64 , (3,3) ,activation='relu' ,padding= 'same' )(us_4)
    bn_4 = BatchNormalization()(conv_4)

    conv_5 = Conv2D(3 , (3, 3) , activation='tanh',padding= 'same' )(bn_4)
    bn_5 = BatchNormalization()(conv_5)

    
    autoencoder = Model(input_img , bn_5)
    autoencoder.compile(optimizer='adam', loss='mae')
    print(autoencoder.summary())

    history = autoencoder.fit(img_list, img_list, batch_size=256 , epochs=10, shuffle=True
            , verbose=1, callbacks=[LR_function])
    
    result_train = autoencoder.evaluate(img_list , img_list)
    print('Train loss:', result_train)

    autoencoder.save('autoencoder_model.h5')

    encoder = Model(input_img , mp_2)
    encoder.save('encoder_model.h5')
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 
    hist_csv_file = 'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    
if(__name__ == '__main__'):
  img_list = load()
  print(img_list.shape)
  model(img_list)
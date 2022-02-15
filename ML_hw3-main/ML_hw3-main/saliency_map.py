#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
from random import randint
import pandas as pd
import numpy as np
import cv2
import keras
from keras import activations
from vis.utils import utils
from vis.visualization import visualize_saliency
from vis.visualization import overlay
import matplotlib.pyplot as plt
import matplotlib.cm as cm

os.environ['KMP_DUPLICATE_LIB_OK']='True'


path = 'train/train'
def load():
    index = randint(0 , 20000)
    image_name = '{:0>5d}.jpg'.format(index)
    image_1 = cv2.imread(os.path.join( path , image_name ) ,cv2.IMREAD_COLOR)
    image_2 = cv2.imread(os.path.join( path , image_name ) ,cv2.IMREAD_GRAYSCALE)
    image_2 = np.array(image_2)
    image_2 = image_2.reshape( 48 , 48 , 1).astype('float')/255
    
    return (image_name , image_1 , image_2)


def saliency_map(model):
    index = utils.find_layer_idx(model , 'dense_5')
    
    model.layers[index].activation = activations.linear
    
    model = utils.apply_modifications(model)
    print("start")
    for i in range(7):
        image_name , image_1 , image_2 = load()
        gradient = visualize_saliency(model , index , filter_indices= None , seed_input= image_2 , backprop_modifier= 'guided')
        
        fig = plt.figure()
        
        ax1 = fig.add_subplot(1 , 3 , 1)
        ax1.set_title(image_name)
        ax1.set_axis_off()
        ax1.imshow(image_1)
        
        ax2 = fig.add_subplot(1 , 3 , 2)
        ax2.set_title('Saliency map')
        ax2.set_axis_off()
        ax2.imshow(gradient , cmap = 'jet')
        
        jet_heatmap = np.uint8(255 * cm.jet(gradient)[... , :3])
        
        ax3 = fig.add_subplot(1 , 3 , 3)
        ax3.set_title('overlay')
        ax3.set_axis_off()
        ax3.imshow(overlay(jet_heatmap , image_1))
        
        plt.show()
    
    return



model = keras.models.load_model('keras_model.h5')
saliency_map(model)


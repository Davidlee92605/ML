import torch 
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from keras.preprocessing.text import Tokenizer
import pickle
def load_labled_training_data(path):
    print("Start loading training data...")
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip('\n').split(' ') for line in lines]
    x_train = [line[2:] for line in lines]
    y_train = [int(line[0]) for line in lines]
    return x_train, y_train

def load_testing_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
       
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
        print(X[:5])
    return X
################################# construct BOW #################################
def word_dictionary(X_train):
    print("Start constructing vocabulary...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    
    vectors = tokenizer.texts_to_matrix(X_train, mode='count')
    print("Vocabulary size: ", vectors[0].shape)
    return vectors

if(__name__ == "__main__"):
    # loading
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print(tokenizer.word_index)
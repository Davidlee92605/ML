import os
import numpy as np
import pandas as pd
import argparse
from gensim.models import word2vec
import torch
import torch.optim as optim
import torch.nn.functional as F

def load_training_data(path):
    if 'training_label' in path:
        with open(path, 'r') as f:
            lines = f.readlines()
 
            lines = [line.strip('\n').split(' ') for line in lines]
        print(lines[:5])
        x_train = [line[2:] for line in lines]
        y_train = [line[0] for line in lines]
        return x_train, y_train
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x_train = [line.strip('\n').split(' ') for line in lines]
        return x_train

def load_testing_data(path='testing_data.txt'):
    with open(path, 'r') as f:
        lines = f.readlines()
       
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        print(len(X))
        X = [sen.split(' ') for sen in X]
        print(len(X))
        print(X[:5])
    return X

def train_word2vec(x):
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model

if __name__ == "__main__":
    print("loading training data ...")
    train_x, y = load_training_data('training_label.txt')
    train_x_no_label = load_training_data('training_nolabel.txt')

    print("loading testing data ...")
    test_x = load_testing_data('testing_data.txt')
    train_x = train_x + train_x_no_label
    #model = train_word2vec(train_x + train_x_no_label + test_x)
    model = train_word2vec(train_x )

    
    print("saving model ...")
    # model.save(os.path.join(path_prefix, 'model/w2v_all.model'))
    model.save('w2v_all.model')
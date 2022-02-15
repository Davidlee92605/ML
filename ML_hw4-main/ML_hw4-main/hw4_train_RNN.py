from torch import nn
from torch.utils.data import Dataset , DataLoader
from gensim.models import Word2Vec
import torch
import os
import sys
import argparse
import numpy as np
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import matplotlib.pyplot as plt

#################################### load model #####################################
def load_training_data(path):
    if 'training_label' in path:
        with open(path, 'r') as f:
            lines = f.readlines()
 
            lines = [line.strip('\n').split(' ') for line in lines]
        x_train = [line[2:] for line in lines]
        y_train = [line[0] for line in lines]
        return x_train, y_train
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x_train = [line.strip('\n').split(' ') for line in lines]
        return x_train

#################################### evaluation #####################################
def evaluation(outputs, labels):
    outputs[outputs>=0.5] = 1
    outputs[outputs<0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

################################### preprocessing ###################################
class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    def get_w2v_model(self):
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self, load=True):
        print("Get embedding ...")
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError

        for i, word in enumerate(self.embedding.wv.vocab):
            print('get words #{}'.format(i+1), end='\r')
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)

        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self):
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)

##################################### load data #####################################
class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True , bidirectional= True)
        self.classifier = nn.Sequential( nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(hidden_dim * 2 , 32),
                                         nn.BatchNorm1d(32) ,
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(32, 4),
                                         nn.BatchNorm1d(4) ,
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(4, 1),
                                         nn.Sigmoid() )
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        x = x[:, -1, :] 
        x = self.classifier(x)
        return x

###################################### training #####################################
def training(x_train, y_train, x_val, y_val, model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))

    loss_values = []
    val_loss_values = []

    batch_size = 256
    epoch = 15
    learning_rate = 0.002

    train_dataset = dataset(X=x_train, y=y_train)
    val_dataset = dataset(X=x_val, y=y_val)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 8)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)

    # training
    model.train()
    train_batch = len(train_loader) 
    validation_batch = len(val_loader) 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_acc = 0
    for k in range(epoch):
        total_loss, total_acc = 0, 0
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(dtype=torch.long)
            labels = labels.to(dtype=torch.float)
            
            optimizer.zero_grad()
            outputs = model(data)
            outputs = outputs.squeeze()
            loss = F.binary_cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            correct = evaluation(outputs, labels)
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	k+1, i+1, train_batch, loss.item(), correct*100/batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/train_batch, total_acc/train_batch*100))
        loss_values.append(total_loss / len(train_dataset))
        # validation
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (data, labels) in enumerate(val_loader):
                data = data.to(dtype=torch.long) 
                labels = labels.to(dtype=torch.float)
                outputs = model(data)
                outputs = outputs.squeeze()
                loss = F.binary_cross_entropy(outputs, labels)
                correct = evaluation(outputs, labels)
                total_acc += (correct / batch_size)
                total_loss += loss.item()
            val_loss_values.append(total_loss / len(val_dataset))
            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/validation_batch, total_acc/validation_batch*100))
            if total_acc > best_acc:
                best_acc = total_acc
                torch.save(model, "ckpt.model")
                print('saving model with acc {:.3f}'.format(total_acc/validation_batch*100))
        print('-----------------------------------------------')
        model.train()
    return loss_values, val_loss_values

###################################### dataset ######################################
class dataset(data.Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = y
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    def __len__(self):
        return len(self.data)

######################################## main #######################################
def main():
    train_path = sys.argv[1]
    w2v_path = 'w2v_all.model'

    sen_len = 20
    fix_embedding = True

    print("loading data ...")
    train_x, y = load_training_data(train_path)
    preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x = preprocess.sentence_word2idx()
    y = preprocess.labels_to_tensor(y)
    model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=50, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
    X_train, X_val, y_train, y_val = train_test_split(train_x, y, test_size = 0.1, random_state = 1, stratify = y)

    loss_values, val_loss_values = training(X_train, y_train, X_val, y_val, model)

    plt.plot(loss_values, '-b', label='training_loss')
    plt.plot(val_loss_values, '-r', label='validation_loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='lower left')
    plt.title("Loss Curve")
    plt.show()

if (__name__ == '__main__'):
    main()
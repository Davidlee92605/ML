import os
import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
import torch
from gensim.models import word2vec
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from torch.utils import data

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Embedding, Dense, Dropout, Activation, Flatten, LSTM, Bidirectional, LeakyReLU, GRU, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger

def load_testing_data(path='testing_data.txt'):
    with open(path, 'r') as f:
        lines = f.readlines()
       
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        print(len(X))
        X = [sen.split(' ') for sen in X]
        print(len(X))
    return X

class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path="w2v.model"):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
    def get_w2v_model(self):
        # 把之前訓練好的 word to vec 模型讀進來
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size
    def add_embedding(self, word):
        # 把 word 加進 embedding，並賦予他一個隨機生成的 representation vector
        # word 只會是 "<PAD>" 或 "<UNK>"
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)
    def make_embedding(self, load=True):
        print("Get embedding ...")
        # 取得訓練好的 Word2vec word embedding
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # 製作一個 word2idx 的 dictionary
        # 製作一個 idx2word 的 list
        # 製作一個 word2vector 的 list
        for i, word in enumerate(self.embedding.wv.vocab):
            print('get words #{}'.format(i+1), end='\r')
            #e.g. self.word2index['he'] = 1 
            #e.g. self.index2word[1] = 'he'
            #e.g. self.vectors[1] = 'he' vector
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # 將 "<PAD>" 跟 "<UNK>" 加進 embedding 裡面
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix
    def pad_sequence(self, sentence):
        # 將每個句子變成一樣的長度
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence
    def sentence_word2idx(self):
        # 把句子裡面的字轉成相對應的 index
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 將每個句子變成一樣的長度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)
    def labels_to_tensor(self, y):
        # 把 labels 轉成 tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)

# class TwitterDataset(data.Dataset):
#     """
#     Expected data shape like:(data_num, data_len)
#     Data can be a list of numpy array or a list of lists
#     input data shape : (data_num, seq_len, feature_dim)
    
#     __len__ will return the number of data
#     """
#     def __init__(self, X, y):
#         self.data = X
#         self.label = y
#     def __getitem__(self, idx):
#         if self.label is None: return self.data[idx]
#         return self.data[idx], self.label[idx]
#     def __len__(self):
#         return len(self.data)

# class LSTM_Net(nn.Module):
#     def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
#         super(LSTM_Net, self).__init__()
#         # 製作 embedding layer
#         self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
#         self.embedding.weight = torch.nn.Parameter(embedding)
#         # 是否將 embedding fix 住，如果 fix_embedding 為 False，在訓練過程中，embedding 也會跟著被訓練
#         self.embedding.weight.requires_grad = False if fix_embedding else True
#         self.embedding_dim = embedding.size(1)
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True , bidirectional= True)
#         self.classifier = nn.Sequential( nn.ReLU(),
#                                          nn.Dropout(dropout),
#                                          nn.Linear(hidden_dim*2 , 32),
#                                          nn.BatchNorm1d(32) ,
#                                          nn.ReLU(),
#                                          nn.Dropout(dropout),
#                                          nn.Linear(32, 4),
#                                          nn.BatchNorm1d(4) ,
#                                          nn.ReLU(),
#                                          nn.Dropout(dropout),
#                                          nn.Linear(4, 1),
#                                          nn.Sigmoid() )
#     def forward(self, inputs):
#         inputs = self.embedding(inputs)
#         x, _ = self.lstm(inputs, None)
#         # x 的 dimension (batch, seq_len, hidden_size)
#         # 取用 LSTM 最後一層的 hidden state
#         x = x[:, -1, :] 
#         x = self.classifier(x)
#         return x

# def testing(batch_size, test_loader, model, device):
#     model.eval()     # 将 model 的模式设为 eval，这样 model 的参数就会被固定住
#     ret_output = []   # 返回的output
#     with torch.no_grad():
#         for i, inputs in enumerate(test_loader):
#             inputs = inputs.to(device, dtype=torch.long)
#             outputs = model(inputs)
#             outputs = outputs.squeeze()
#             outputs[outputs>=0.5] = 1 # 大于等于0.5为正面
#             outputs[outputs<0.5] = 0 # 小于0.5为负面
#             ret_output += outputs.int().tolist()
    
#     return ret_output

def keras_test(x_test):
    model = keras.models.load_model('keras_model.h5')
    y_predict = model.predict(x_test)

    return y_predict

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading testing data ...")
    test_x = load_testing_data('testing_data.txt')
    w2v_path = 'w2v_all.model'
    sen_len = 20
    fix_embedding = True # fix embedding during training
    batch_size = 128
    epoch = 5
    lr = 0.001

    # 对test_x作预处理
    preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    test_x = preprocess.sentence_word2idx()

    Y_preds_label = keras_test(test_x.numpy())
    print(Y_preds_label.shape)
    
    ans = []
    for i in range(Y_preds_label.shape[0]):
        sum = 0
        for j in range(Y_preds_label.shape[1]):
            sum += Y_preds_label[i][j]
        ans.append(sum / Y_preds_label.shape[1])

        if(ans[i]>=0.5):
            ans[i] = 1
        else:
            ans[i] = 0
    print(ans[:10])
    #test_dataset = TwitterDataset(X=test_x, y=None)
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)

    # # 读取模型
    # print('\nload model ...')
    # model = torch.load('ckpt.model')
    # # 测试模型
    # outputs = testing(batch_size, test_loader, model, device)

    # # 保存为 csv 
    tmp = pd.DataFrame({"id":[str(i) for i in range(len(ans))],"label":ans})
    print("save csv ...")
    tmp.to_csv('predict.csv', index=False)
    print("Finish Predicting")

if(__name__ == "__main__"):
    main()
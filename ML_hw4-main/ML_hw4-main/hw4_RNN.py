import os
import torch
import argparse
import numpy as np
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

# preprocess.py
# 這個 block 用來做 data 的預處理
def load_training_data(path):
    if 'training_label' in path:
        with open(path, 'r') as f:
            lines = f.readlines()
 
            lines = [line.strip('\n').split(' ') for line in lines]
        #print(lines[:5])
        x_train = [line[2:] for line in lines]
        y_train = [line[0] for line in lines]
        return x_train, y_train
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x_train = [line.strip('\n').split(' ') for line in lines]
        return x_train

def evaluation(outputs, labels):
    # outputs => probability (float)
    # labels => labels
    outputs[outputs>=0.5] = 1 # 大於等於 0.5 為有惡意
    outputs[outputs<0.5] = 0 # 小於 0.5 為無惡意
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

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

############################  Dataset ############################
class TwitterDataset(data.Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)
    
    __len__ will return the number of data
    """
    def __init__(self, X, y):
        self.data = X
        self.label = y
    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]
    def __len__(self):
        return len(self.data)

############################   LSTM   ############################
class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        # 製作 embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否將 embedding fix 住，如果 fix_embedding 為 False，在訓練過程中，embedding 也會跟著被訓練
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True , bidirectional= True)
        self.classifier = nn.Sequential( nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(hidden_dim*2 , 32),
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
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最後一層的 hidden state
        x = x[:, -1, :] 
        x = self.classifier(x)
        return x

############################   Train   ###########################
def training(batch_size, n_epoch, lr, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    
    model.train() # 將 model 的模式設為 train，這樣 optimizer 就可以更新 model 的參數
    #criterion = nn.BCELoss() # 定義損失函數，這裡我們使用 binary cross entropy loss
    t_batch = len(train) 
    v_batch = len(valid) 
    #optimizer = optim.Adam(model.parameters(), lr=lr) # 將模型的參數給 optimizer，並給予適當的 learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr ) 
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # 這段做 training
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(dtype=torch.long) # device 為 "cuda"，將 inputs 轉成 torch.cuda.LongTensor
            labels = labels.to(dtype=torch.float) # device為 "cuda"，將 labels 轉成 torch.cuda.FloatTensor，因為等等要餵進 criterion，所以型態要是 float
            optimizer.zero_grad() # 由於 loss.backward() 的 gradient 會累加，所以每次餵完一個 batch 後需要歸零
            outputs = model(inputs) # 將 input 餵給模型

            outputs = outputs.squeeze() # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion()
            loss = F.binary_cross_entropy(outputs , labels) # 計算此時模型的 training loss

            loss.backward() # 算 loss 的 gradient
            optimizer.step() # 更新訓練模型的參數

            # zero grad before new step
            optimizer.zero_grad()

            correct = evaluation(outputs, labels) # 計算此時模型的 training accuracy
            total_acc += (correct / batch_size)
            total_loss += loss.item()

            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))

        # 這段做 validation
        model.eval() # 將 model 的模式設為 eval，這樣 model 的參數就會固定住
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(dtype=torch.long) # device 為 "cuda"，將 inputs 轉成 torch.cuda.LongTensor
                labels = labels.to(dtype=torch.float) # device 為 "cuda"，將 labels 轉成 torch.cuda.FloatTensor，因為等等要餵進 criterion，所以型態要是 float

                outputs = model(inputs) # 將 input 餵給模型
                outputs = outputs.squeeze() # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion()
                loss = F.binary_cross_entropy(outputs , labels) # 計算此時模型的 validation loss
                
                correct = evaluation(outputs, labels) # 計算此時模型的 validation accuracy
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                # 如果 validation 的結果優於之前所有的結果，就把當下的模型存下來以備之後做預測時使用
                best_acc = total_acc
                #torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
                torch.save(model, "ckpt.model")
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
        print('-----------------------------------------------')
        model.train() # 將 model 的模式設為 train，這樣 optimizer 就可以更新 model 的參數（因為剛剛轉成 eval 模式）

############################  Keras Train   ###########################
def keras_train(embedding, X_train, X_val, y_train, y_val):
    print(type(y_train))
    print(type(X_train))
    print(type(X_val))
    print(type(y_val))
    embedding_layer = Embedding(input_dim=embedding.shape[0],
                            output_dim=embedding.shape[1],
                            weights=[embedding],
                            trainable=False )
    
    lrate = ReduceLROnPlateau(monitor='val_accuracy', 
                                factor=0.2, 
                                patience=2, 
                                verbose=1, 
                                min_lr=0.00000001)
    earlystopping = EarlyStopping(monitor='val_accuracy', 
                                    patience=10, 
                                    verbose=1, 
                                    mode='auto')
    
    drop_rate =0.4
    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(256, activation='tanh', kernel_initializer='Orthogonal', recurrent_initializer='Orthogonal', return_sequences=True, unit_forget_bias=False, dropout=drop_rate, recurrent_dropout=drop_rate), merge_mode='sum'))
    #model.add(Bidirectional(LSTM(256, activation='tanh', kernel_initializer='Orthogonal', recurrent_initializer='Orthogonal', return_sequences=False, unit_forget_bias=False, dropout=drop_rate, recurrent_dropout=drop_rate), merge_mode='sum'))
    model.add(Dense(256))
    model.add(Dropout(drop_rate))
    model.add(LeakyReLU(0.2))
    model.add(Dense(64))
    model.add(Dropout(drop_rate))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    adam = optimizers.Adam(lr=0.002, clipvalue=1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()


    model.fit(x=X_train, y=y_train, batch_size=512, epochs=5, validation_data=(X_val, y_val), 
            callbacks=[ lrate, earlystopping], shuffle=True)

    result_train = model.evaluate(X_train,y_train)
    print('Train Acc:', result_train[1])
    result_val = model.evaluate(X_val,y_val)
    print('Val Acc:', result_val[1])

    model.save('keras_model.h5')



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 處理好各個 data 的路徑
    train_with_label = 'training_label.txt'
    train_no_label = 'training_nolabel.txt'
    testing_data = 'testing_data.txt'

    w2v_path = 'w2v_all.model' # 處理 word to vec model 的路徑

    # 定義句子長度、要不要固定 embedding、batch 大小、要訓練幾個 epoch、learning rate 的值、model 的資料夾路徑
    sen_len = 20
    fix_embedding = True # fix embedding during training
    batch_size = 128
    epoch = 10
    lr = 0.002


    print("loading data ...") # 把 'training_label.txt' 跟 'training_nolabel.txt' 讀進來
    train_x, y = load_training_data(train_with_label)
    #print('train_x',train_x[:10])

    # 對 input 跟 labels 做預處理
    preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x = preprocess.sentence_word2idx()
    y = preprocess.labels_to_tensor(y)


    # 製作一個 model 的對象
    # model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
    # model = model.to(device) 
    # # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）

    # 把 data 分为 training data 和 validation data（将一部分 training data 作为 validation data）
    X_train, X_val, y_train, y_val = train_test_split(train_x, y, test_size = 0.1, random_state = 1, stratify = y)
    # print('Train | Len:{} \nValid | Len:{}'.format(len(y_train), len(y_val)))
    # #input('stop')
    # #print(type(X_train))
    # #input('stop')
    # # 把 data 做成 dataset 供 dataloader 取用
    # print(type(X_train.numpy()))
    # print(type(y_train.numpy()))
    # train_dataset = TwitterDataset(X=X_train, y=y_train)
    # val_dataset = TwitterDataset(X=X_val, y=y_val)
    # #####################

    # # 把 data 轉成 batch of tensors
    # train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
    #                                             batch_size = batch_size,
    #                                             shuffle = True,
    #                                             num_workers = 8)

    # val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
    #                                             batch_size = batch_size,
    #                                             shuffle = False,
    #                                             num_workers = 8)
    # #################

    # # 開始訓練
    # training(batch_size, epoch, lr , train_loader, val_loader, model, device)
    keras_train(embedding , X_train.numpy() , X_val.numpy() , y_train.numpy() , y_val.numpy())

if(__name__ == '__main__'):
    main()
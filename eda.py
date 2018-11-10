# encoding: utf-8
"""
@author: pkusp
@contact: pkusp@outlook.com

@version: 1.0
@file: eda.py
@time: 2018/11/10 下午3:51

这一行开始写关于本文件的说明与解释
"""
'''
pipline:
观察正负例数量
空缺值
拆分训练集、验证集
使用不同的embeddings(glove,word2vec,自己训练)训练后，进行融合
    Keras embedding层类似word2vec，将输入的二维(用字典id表示的句子)转化为三维张量(即将id训练成vec)
f1score评价

LSTM + CNN + attention

tokenizer用来将query转化为序列(先遍历得到字典，然后按频率排序得到id)

'''


import time
import numpy as np
import pandas as pd
from config import input_path

t_start = time.time()

import os
print(os.listdir("../input")) # kernel only比赛，数据存储路径
print("welcome!")

from tqdm import tqdm  # 进度条工具
import math
from sklearn.model_selection import train_test_split


train_df = pd.read_csv("../input/train.csv")  # kernel only比赛，数据存储路径
train_df, val_df = train_test_split(train_df, test_size=0.1)  # 拆分数据集为训练集+验证集

pos_x_df = train_df[train_df["target"]==1]
neg_x_df = train_df[train_df["target"]==0]
print("pos/neg data:",len(pos_x_df)/10000,len(neg_x_df)/10000)  # 正负例比为 1：15 需要sample

for t in pos_x_df[:10]["question_text"]:
    print(t)


#  tokenizes使用:https://blog.csdn.net/zzulp/article/details/76146947

## some config values
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

## fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)  # 向量维数
tokenizer.fit_on_texts(list(train_X))  # 用所有问题text训练向量
train_X = tokenizer.texts_to_sequences(train_X)  # 将单词用字典id表示
val_X = tokenizer.texts_to_sequences(val_X)  # 将句子中的单词用字典的序号表示
test_X = tokenizer.texts_to_sequences(test_X)  # 将单词用字典序号表示

## Pad the sentences
train_X = pad_sequences(train_X, maxlen=maxlen)  # 补全长度
val_X = pad_sequences(val_X, maxlen=maxlen)  # 补全长度
test_X = pad_sequences(test_X, maxlen=maxlen)  # 补全长度

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)  # embedding层，用来训练词向量
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
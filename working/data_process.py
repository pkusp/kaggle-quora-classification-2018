
import time
import numpy as np 
import pandas as pd
from config import input_path


t_start = time.time()

import os
print(os.listdir("../input")) # kernel only比赛，数据存储路径
print("welcome!")

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os
from tqdm import tqdm  # 进度条工具
import math
from sklearn.model_selection import train_test_split


train_df = pd.read_csv("../input/train.csv")  # kernel only比赛，数据存储路径
train_df, val_df = train_test_split(train_df, test_size=0.1)  # 拆分数据集为训练集+验证集

# Source https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# 读取glove词向量，词为key，向量为value的字典,参照Keras文档
embeddings_index = {}
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
for line in tqdm(f):  # 进度条展示方法
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# # Convert values to embeddings
# # 将一句话转化为三维张量，即[[句子]，[[词vec],[[词vec]],...],[句子],...]
# def text_to_array(text):
#     empyt_emb = np.zeros(300)  # 初始化默认向量为全零
#     text = text[:-1].split()[:30]  # 句子截断长度为30
#     embeds = [embeddings_index.get(x, empyt_emb) for x in text]  # dict.get(x,y) = defaultdict[x],其中默认值为empyt_emb
#     embeds+= [empyt_emb] * (30 - len(embeds))  # 长度不满30的用默认零数组empty_emb补全
#     return np.array(embeds)  # 三维张量

# train_vects = [text_to_array(X_text) for X_text in tqdm(train_df["question_text"])]
# 转化验证集为张量表示，结果为三维张量的数组
val_vects = np.array([text_to_array(X_text) for X_text in tqdm(val_df["question_text"][:3000])])
# 验证集的label，取前3000个
val_y = np.array(val_df["target"][:3000])

# # Data providers
# batch_size = 128
# # 生成每个batch的训练集
# def train_batch_gen(train_df)->tuple:
#     n_batches = math.ceil(len(train_df) / batch_size)  # ceil为上取整，求batch数量
#     while True: 
#         train_df = train_df.sample(frac=1.)  # Shuffle the data. 打乱训练集顺序（100%采样）
#         for i in range(n_batches):  # 从训练集中顺序取每个batch数据，句子在第一列
#             texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]
#             text_arr = np.array([text_to_array(text) for text in texts])  # [query转化为三维张量]
#             yield text_arr, np.array( train_df["target"][i*batch_size:(i+1)*batch_size] )  # 每次yield一个batch的train和对应的label

# # prediction par
# batch_size = 256
# # 生成每个batch的测试集
# def test_batch_gen(test_df):
#     n_batches = math.ceil(len(test_df) / batch_size)
#     for i in range(n_batches):
#         texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]
#         text_arr = np.array([text_to_array(text) for text in texts])
#         yield text_arr

test_df = pd.read_csv("../input/test.csv")


class DataProcess(object):
    def __init__(self,train_df=None,test_df=None,batch_size=128):
        self.train_df=train_df
        self.test_df=test_df
        self.batch_size=batch_size
        
    def text_to_array(self,text):
        empyt_emb = np.zeros(300)  # 初始化默认向量为全零
        text = text[:-1].split()[:30]  # 句子截断长度为30
        embeds = [embeddings_index.get(x, empyt_emb) for x in text]  # dict.get(x,y) = defaultdict[x],其中默认值为empyt_emb
        embeds+= [empyt_emb] * (30 - len(embeds))  # 长度不满30的用默认零数组empty_emb补全
        return np.array(embeds)  # 三维张量        

    def train_batch_gen(self,train_df)->tuple:
        batch_size=self.batch_size
        n_batches = math.ceil(len(train_df) / batch_size)  # ceil为上取整，求batch数量
        while True: 
            train_df = train_df.sample(frac=1.)  # Shuffle the data. 打乱训练集顺序（100%采样）
            for i in range(n_batches):  # 从训练集中顺序取每个batch数据，句子在第一列
                texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]
                text_arr = np.array([self.text_to_array(text) for text in texts])  # [query转化为三维张量]
                yield text_arr, np.array( train_df["target"][i*batch_size:(i+1)*batch_size] )  # 每次yield一个batch的train和对应的label

    # 生成每个batch的测试集
    def test_batch_gen(self,test_df):
        n_batches = math.ceil(len(test_df) / batch_size)
        for i in range(n_batches):
            texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]
            text_arr = np.array([text_to_array(text) for text in texts])
            yield text_arr


dprocess=DataProcess(train_df=train_df,batch_size=128)

# encoding: utf-8
"""
@author: pkusp
@contact: pkusp@outlook.com

@version: 1.0
@file: bi-lstm_baseline.py
@time: 2018/11/20 下午4:43

LSTM is all you need
"""
# 2000 seconds
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import time
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Bidirectional

t_start = time.time()
# read train data
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_df, val_df = train_test_split(train_df, test_size=0.1)
# Data providers
train_batch_size = 128
test_batch_size = 256


def embeddings_to_dict():
    """
    # embdedding setup
    # Source https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    # 读取glove的embeddings进入embeddings_index，形式为「word:vec」
    """
    embeddings_index = {}
    f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
    for line in tqdm(f):
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def text_to_vec(text):
    """
    :param text: 输入的一句话
    :return: 这句话的二维vector形式
    # Convert values to embeddings
    # 将文本转化为向量，每次转化一句text，返回结果为二维数组形式「「Word vec」,「word vec」,...」
    """
    empyt_emb = np.zeros(300)  # embeddings向量维度为300，初始化为300
    text = text[:-1].split()[:30]  # 去query的前30个词
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]  # 将text内每个单词转化为vec，如果没有这个单词则填0(empty_emb)
    embeds += [empyt_emb] * (30 - len(embeds))  # text长度小于30的进行padding
    return np.array(embeds)  # 返回text的二维数组表示


def train_batch_gen(train_df):
    """
    :param train_df:
    :return:
    # 生成训练集batch的函数
    # 生成一个batch三维张量训练集的vec，以及相应batch的label
    """

    n_batches = int(math.ceil(len(train_df) / train_batch_size))  # 计算batch数量
    while True:
        train_df = train_df.sample(frac=1.)  # Shuffle the data. 100%采样，相当于打乱训练集数据
        for i in range(n_batches):
            texts = train_df.iloc[i * train_batch_size:(i + 1) * train_batch_size, 1]  # text在第一列
            text_vec_arr = np.array([text_to_vec(text) for text in texts])
            yield text_vec_arr, np.array(train_df["target"][i * train_batch_size:(i + 1) * train_batch_size])


def test_batch_gen(test_df):
    """
    :param test_df:
    :return:
    # 生成测试集的batch，只返回该batch训练集的输入，格式为三维张量
    """
    n_batches = int(math.ceil(len(test_df) / test_batch_size))
    for i in range(n_batches):
        texts = test_df.iloc[i * test_batch_size:(i + 1) * test_batch_size, 1]
        text_vec_arr = np.array([text_to_vec(text) for text in texts])
        yield text_vec_arr


embeddings_index = embeddings_to_dict()

# 将验证集的二维输入的text转化为三维的张量形式，取前3000
val_vects = np.array([text_to_vec(X_text) for X_text in tqdm(val_df["question_text"][:3000])])
val_y = np.array(val_df["target"][:3000])

# 构造LSTM网络
model = Sequential()
model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True),
                        input_shape=(30, 300)))
model.add(Bidirectional(CuDNNLSTM(64)))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 训练网络
mg = train_batch_gen(train_df)  # mg是一个生成器，每次生成一个batch的数据 训练集batch输入的三维张量，训练集batch的label
model.fit_generator(mg, epochs=20, steps_per_epoch=1000,
                    validation_data=(val_vects, val_y),  # 没词迭代的验证数据
                    verbose=True)
# prediction part
# 使用训练好的模型进行预测
# 每次预测一个batch，将该batch的预测结果extend进最终结果
all_preds = []  # 预测的结果
for batch_x in tqdm(test_batch_gen(test_df)):  # 每次生成一个batch数据
    batch_pred = model.predict(batch_x).flatten()
    all_preds.extend(batch_pred)

# 预测结果为【0，1】的值，设定阈值(0.5)将其转化为0，1
y_te = (np.array(all_preds) > 0.5).astype(np.int)

# 保存结果
submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})
submit_df.to_csv("submission-{}.csv".format(str(datetime.now().date())), index=False)

t_end = time.time()
print("run{}s".format(t_end - t_start))

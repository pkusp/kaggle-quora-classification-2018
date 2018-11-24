
# ################################ LSTM attention baseline ############################################################
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split


SEQ_LEN = 100  # magic number - length to truncate sequences of words

train_df = pd.read_csv("../input/train.csv")
train_df, val_df = train_test_split(train_df, test_size=0.07)

#minor eda: average question length (in words) is 12  , majority are under 12 words
train_df.question_text.str.split().str.len().describe()

### Unclear why fails to open [encoding error], format is same as for glove. Will Debug, Dan:
### f = open('../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt')

# embedding setup
# Source https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
#
embeddings_index = {}
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
# f = open('../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt')
for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


import re
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")
STOP_WORDS = "\" \' [ ] . , ! : ; ?".split(" ")
def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
        # return [w.lower() for w in words if w not in stop_words and w != '' and w != ' ']
    return [w.lower() for w in words if w != '' and w != ' ']



# Convert values to embeddings
def text_to_array(text):
    empyt_emb = np.zeros(300)
    text = basic_tokenizer(text[:-1])[:SEQ_LEN]
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    embeds+= [empyt_emb] * (SEQ_LEN - len(embeds))
    return np.array(embeds)

# train_vects = [text_to_array(X_text) for X_text in tqdm(train_df["question_text"])]
val_vects = np.array([text_to_array(X_text) for X_text in tqdm(val_df["question_text"][:3000])])
val_y = np.array(val_df["target"][:3000])



# Data providers
batch_size = 256

def batch_gen(train_df):
    n_batches = math.ceil(len(train_df) / batch_size)
    while True:
        train_df = train_df.sample(frac=1.)  # Shuffle the data.
        for i in range(n_batches):
            texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]
            text_arr = np.array([text_to_array(text) for text in texts])
            yield text_arr, np.array(train_df["target"][i*batch_size:(i+1)*batch_size])


from keras.models import Sequential,Model
from keras.layers import CuDNNLSTM, Dense, Bidirectional, Input,Dropout

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


# https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


inp = Input(shape=(SEQ_LEN,300 ))
x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(inp)
x = Bidirectional(CuDNNLSTM(64,return_sequences=True))(x)
x = Attention(SEQ_LEN)(x)
x = Dense(256, activation="relu")(x)
# x = Dropout(0.25)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


mg = batch_gen(train_df)
model.fit_generator(mg, epochs=20,
                    steps_per_epoch=1000,
                    validation_data=(val_vects, val_y),
                    verbose=True)


# prediction part
batch_size = 256
def batch_gen(test_df):
    n_batches = math.ceil(len(test_df) / batch_size)
    for i in range(n_batches):
        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]
        text_arr = np.array([text_to_array(text) for text in texts])
        yield text_arr

test_df = pd.read_csv("../input/test.csv")

all_preds = []
for x in tqdm(batch_gen(test_df)):
    all_preds.extend(model.predict(x).flatten())


y_te = (np.array(all_preds) > 0.35).astype(np.int)

submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})
submit_df.to_csv("submission.csv", index=False)


###################################################################################################################3



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
from keras.models import Sequential,Model
from keras.layers import CuDNNLSTM, Dense, Bidirectional, Input,Dropout

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


t_start = time.time()
# read train data
train_df = train_clean
test_df = test_clean
train_df, val_df = train_test_split(train_df, test_size=0.1)
# Data providers
train_batch_size = 128
test_batch_size = 256
SEQ_LEN = 50

# train_df EDA
train_df.question_text.str.split().str.len().describe()


# https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


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
    text = text[:-1].split()[:SEQ_LEN]  # 取query的前30个词
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]  # 将text内每个单词转化为vec，如果没有这个单词则填0(empty_emb)
    embeds += [empyt_emb] * (SEQ_LEN - len(embeds))  # text长度小于30的进行padding
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


#
# inp = Input(shape=(SEQ_LEN,300 ))
# x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(inp)
# x = Bidirectional(CuDNNLSTM(64,return_sequences=True))(x)
# x = Attention(SEQ_LEN)(x)
# x = Dense(256, activation="relu")(x)
# # x = Dropout(0.25)(x)
# x = Dense(1, activation="sigmoid")(x)
# model = Model(inputs=inp, outputs=x)
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# 构造LSTM网络
model = Sequential()
model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True),input_shape=(SEQ_LEN, 300)))
model.add(Bidirectional(CuDNNLSTM(64,return_sequences=True)))
model.add(Attention(SEQ_LEN))
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 训练网络
mg = train_batch_gen(train_df)  # mg是一个生成器，每次生成一个batch的数据 训练集batch输入的三维张量，训练集batch的label
model.fit_generator(mg, epochs=20,
                    steps_per_epoch=1000,
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
submit_df.to_csv("submission.csv", index=False)

t_end = time.time()
print("run{}s".format(t_end - t_start))



### 一、[How to: preprocessing when using embeddings](https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings)
```python
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
# read data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)

# 统计训练集中所有词的词频，返回结果为字典『word:count』
def build_vocab(sentences, verbose =  True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

# vocab为词频字典
sentences = train["question_text"].progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)
print({k: vocab[k] for k in list(vocab)[:5]})

# 利用KeyedVectors方法读取预训练的embeddings, embeddings_index为读取的字典->「word:vector」
from gensim.models import KeyedVectors
news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)
print(embeddings_index["hello"])

# 检查训练集的词表和预训练的embeddings单词的交集
import operator
def check_coverage(vocab,embeddings_index):
    catch_mp = {}
    oov = {}
    count_catch = 0
    cnt_miss = 0
    for word in tqdm(vocab):
        try:
            catch_mp[word] = embeddings_index[word]
            count_catch += vocab[word]
        except:

            oov[word] = vocab[word]
            cnt_miss += vocab[word]
            pass
	# 覆盖到的词典的比例 24%
    print('Found embeddings for {:.2%} of vocab'.format(len(catch_mp) / len(vocab)))
    # 覆盖到的总词数 78%
    print('Found embeddings for  {:.2%} of all text'.format(count_catch / (count_catch + cnt_miss)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

# 预训练的embeddings覆盖率太低，很多训练集中的词没有embedding！
oov = check_coverage(vocab,embeddings_index)

# 看看top10 out of vocab的词是哪些：
oov[:10]
# On first place there is "to". Why? Simply because "to" was removed when the GoogleNews Embeddings were trained.
'''
[('to', 403183),
 ('a', 402682),
 ('of', 330825),
 ('and', 251973),
 ('India?', 16384),
 ('it?', 12900),
 ('do?', 8753),
 ('life?', 7753),
 ('you?', 6295),
 ('me?', 6202)]
 '''

# for now we take care about the splitting of punctuation as this also seems to be a Problem.
# But what do we do with the punctuation then - Do we want to delete or consider as a token?
# I would say: It depends. If the token has an embedding, keep it, if it doesn't we don't need it anymore. So lets check:
'?' in embeddings_index  # false
'&' in embeddings_index # true

# Interesting. While "&" is in the Google News Embeddings, "?" is not.
# So we basically define a function that splits off "&" and removes other punctuation.
def clean_text(x):
    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x

# 在读取训练集query时加入clean_text()，以去除标点
train["question_text"] = train["question_text"].progress_apply(lambda x: clean_text(x))
sentences = train["question_text"].apply(lambda x: x.split())
vocab = build_vocab(sentences)

# 去除标点后检查训练集与embeddings的重合率：由24% 78% -> 57% 90%
oov = check_coverage(vocab,embeddings_index)

# 再次的，检查漏掉的top10单词是哪些：（大部分都是数字！）
oov[:10]
'''
[('to', 406298),
 ('a', 403852),
 ('of', 332964),
 ('and', 254081),
 ('2017', 8781),
 ('2018', 7373),
 ('10', 6642),
 ('12', 3694),
 ('20', 2942),
 ('100', 2883)]
 '''

# 现在检查top10的embeddings看看是否有所发现
for i in range(10):
    print(embeddings_index.index2entity[i])
'''
</s>
in
for
that
is
on
##
The
with
said
'''
# hmm why is "##" in there? Simply because as a reprocessing all numbers bigger tha 9 have been replaced by hashs.
# I.e. 15 becomes ## while 123 becomes ### or 15.80€ becomes ##.##€.
# So lets mimic this preprocessing step to further improve our embeddings coverage

import re
# 将数字替换为#号，按长度来替换
def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

# 在读取训练集时同时clean_number()
train["question_text"] = train["question_text"].progress_apply(lambda x: clean_numbers(x))
sentences = train["question_text"].progress_apply(lambda x: x.split())
vocab = build_vocab(sentences)

# 再次的，看看处理后的embeddings和训练集的交集：57% 90% -> 60% 90%
oov = check_coverage(vocab,embeddings_index)

# 继续看看top20漏掉的词
oov[:20]

# Ok now we take care of common misspellings when using american/british vocab and replacing a few "modern" words with "social media" for this task
# I use a multi regex script I found some time ago on stack overflow.
# Additionally we will simply remove the words "a","to","and" and "of" since those have obviously been downsampled when training the GoogleNews Embeddings.

'''
[('to', 406298),
 ('a', 403852),
 ('of', 332964),
 ('and', 254081),
 ('favourite', 1247),
 ('bitcoin', 987),
 ('colour', 976),
 ('doesnt', 918),
 ('centre', 886),
 ('Quorans', 858),
 ('cryptocurrency', 822),
 ('Snapchat', 807),
 ('travelling', 705),
 ('counselling', 634),
 ('btech', 632),
 ('didnt', 600),
 ('Brexit', 493),
 ('cryptocurrencies', 481),
 ('blockchain', 474),
 ('behaviour', 468)]
 '''

# 从上面可知，标点相关的通用问题解决差不多了，现在开始处理拼写的问题
# 将字典编译成正则表达式
def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

# 手写常见的拼写错误
mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium'

                }
mispellings, mispellings_re = _get_mispell(mispell_dict)

# 闭包处理, 将所有的错误拼写替换为标准拼写
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)

# 再次的，将replace_typical_misspell()加入到训练集的读取过程，使拼写更加标准化，并去除一些embeddings漏掉的top几的词
train["question_text"] = train["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
sentences = train["question_text"].progress_apply(lambda x: x.split())
to_remove = ['a','to','of','and']
sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]
vocab = build_vocab(sentences)

# 再次检查embeddings和训练集的交集：60% 90% -> 60% 99% 分别为（字典遗漏和总词数遗漏）
oov = check_coverage(vocab,embeddings_index)

# 最后检查遗漏的top20，都是新词，没问题了
oov[:20]
'''
[('bitcoin', 987),
 ('Quorans', 858),
 ('cryptocurrency', 822),
 ('Snapchat', 807),
 ('btech', 632),
 ('Brexit', 493),
 ('cryptocurrencies', 481),
 ('blockchain', 474),
 ('behaviour', 468),
 ('upvotes', 432),
 ('programme', 402),
 ('Redmi', 379),
 ('realise', 371),
 ('defence', 364),
 ('KVPY', 349),
 ('Paytm', 334),
 ('grey', 299),
 ('mtech', 281),
 ('Btech', 262),
 ('bitcoins', 254)]
 '''

```
### 二、[LSTM is all you need](https://www.kaggle.com/mihaskalic/lstm-is-all-you-need-well-maybe-embeddings-also)
```python
# bi-LSTM baseline
# encoding: utf-8
"""
@author: pkusp
@contact: pkusp@outlook.com

@version: 1.0
@file: tmp.py
@time: 2018/11/23 下午4:48

这一行开始写关于本文件的说明与解释
"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split

# read train data
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

train_df, val_df = train_test_split(train_df, test_size=0.1)

# embdedding setup
# Source https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# 读取glove的embeddings进入embeddings_index，形式为「word:vec」
embeddings_index = {}
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# Convert values to embeddings
# 将文本转化为向量，每次转化一句text，返回结果为二维数组形式「「Word vec」,「word vec」,...」
def text_to_vec(text):
    empyt_emb = np.zeros(300)  # embeddings向量维度为300，初始化为300
    text = text[:-1].split()[:30]  # 去query的前30个词
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]  # 将text内每个单词转化为vec，如果没有这个单词则填0(empty_emb)
    embeds += [empyt_emb] * (30 - len(embeds))  # text长度小于30的进行padding
    return np.array(embeds)  # 返回text的二维数组表示


# 将验证集的二维输入的text转化为三维的张量形式，取前3000
# train_vects = [text_to_array(X_text) for X_text in tqdm(train_df["question_text"])]
val_vects = np.array([text_to_vec(X_text) for X_text in tqdm(val_df["question_text"][:3000])])
val_y = np.array(val_df["target"][:3000])


# 生成训练集的batch函数
# Data providers
train_batch_size = 128

# 生成一个batch三维张量训练集的vec，以及相应batch的label
def batch_gen(train_df):
    n_batches = int(math.ceil(len(train_df) / train_batch_size))  # 计算batch数量
    while True:
        train_df = train_df.sample(frac=1.)  # Shuffle the data. 100%采样，相当于打乱训练集数据
        for i in range(n_batches):
            texts = train_df.iloc[i*train_batch_size:(i+1)*train_batch_size, 1]  # text在第一列
            text_vec_arr = np.array([text_to_vec(text) for text in texts])
            yield text_vec_arr, np.array(train_df["target"][i*train_batch_size:(i+1)*train_batch_size])


# 导入Keras的网络
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Bidirectional


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
mg = batch_gen(train_df)  # mg是一个生成器，每次生成一个batch的数据 训练集batch输入的三维张量，训练集batch的label
model.fit_generator(mg, epochs=20,
                    steps_per_epoch=1000,
                    validation_data=(val_vects, val_y),  # 没词迭代的验证数据
                    verbose=True)


# prediction part
# 使用训练好的模型进行预测
test_batch_size = 256


# 生成测试集的batch，只返回该batch训练集的输入，格式为三维张量
def test_batch_gen(test_df):
    n_batches = int(math.ceil(len(test_df) / test_batch_size))
    for i in range(n_batches):
        texts = test_df.iloc[i*test_batch_size:(i+1)*test_batch_size, 1]
        text_vec_arr = np.array([text_to_vec(text) for text in texts])
        yield text_vec_arr


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

```
### 三、[A look at different embeddings](https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings)
```python
# try different embeddings
```

### 四、[LSTM Attention baseline](https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb)
引用的[Attention](https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043)
```python
attention
```

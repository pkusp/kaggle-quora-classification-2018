# encoding: utf-8
"""
@author: pkusp
@contact: pkusp@outlook.com

@version: 1.0
@file: lstm_cnn_attention_kernel.py
@time: 2018/11/10 下午5:44

这一行开始写关于本文件的说明与解释
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from nltk.tokenize import TweetTokenizer
import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
pd.set_option('max_colwidth',400)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from sklearn.preprocessing import OneHotEncoder

import os
print(os.listdir("../input/embeddings/glove.840B.300d/"))
# read dataset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sub = pd.read_csv('../input/sample_submission.csv')
#
import os
print('Available embeddings:', os.listdir("../input/embeddings/"))
# 列值统计：
# 0    1225312
# 1      80810
# Name: target, dtype: int64
# We have a seriuos disbalance - only ~6% of data are positive. No wonder the metric for the competition is f1-score.
# precision = TP / (TP + FP)
# recall = TP / (TP + FN)
# accuracy = (TP + TN) / (TP + FP + TN + FN)
# F1 Score = 2*P*R/(P+R)，其中P和R分别为 precision 和 recall
train["target"].value_counts()


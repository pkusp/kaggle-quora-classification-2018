> 说明：本项目为kaggle比赛内容，开始时间2018.11.09

# kaggle - Quora insincere classification
- 比赛链接:[Quora Insincere Classification](https://www.kaggle.com/c/quora-insincere-questions-classification)
- 比赛形式: *kernel only*


## EDA

```python
import os
print(os.listdir("../input/embeddings/glove.840B.300d/"))
# read dataset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sub = pd.read_csv('../input/sample_submission.csv')
--
Train shape :  (1306122, 3)
Test shape :  (56370, 2)
--
train columns = Index(['qid', 'question_text', 'target'], dtype='object')
train["target"].value_counts()
```
```python
 0    1225312
 1      80810
 Name: target, dtype: int64
```
- 正负列比约为`1：15`，故采用`F1 score`


> $F1-score = \frac{2*(P*RP)}{P+R}$,其中P和R分别为 precision 和 recall


```python
precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = (TP + TN) / (TP + FP + TN + FN)
```

## 一些操作技巧整理

- `KeyeVectors`读取预训练的词向量

```python
import gensim
from gensim.models import KeyedVectors

word2vec_model_path = './data/data_vec.txt' ##词向量文件的位置
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=False,unicode_errors='ignore')
word2vec_dict = {}
for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
    if '.bin' not in word2vec_model_path:
        word2vec_dict[word] = vector
    else:
        word2vec_dict[word] = vector /np.linalg.norm(vector) 
for each in word2vec_dict:
    print (each,word2vec_dict[each])
# --------------------- 
# 原文：https://blog.csdn.net/yangfengling1023/article/details/81705109 
```
- use `tqdm` to see progress bar
```python
import tqdm
```
- split句子的简洁方法
```python
sentences = train["question_text"].progress_apply(lambda x: x.split()).values

```



## 记录
- 观察正负例数量
- 空缺值
- 拆分训练集、验证集
- 使用不同的`embeddings`(glove,word2vec,自己训练)训练后，进行融合
    - `Keras-embedding`层类似word2vec，将输入的二维(用字典id表示的句子)转化为三维张量(即将id训练成vec)
- f1score评价

- LSTM + CNN + attention

- `tokenizer` 用来将query转化为序列(先遍历得到字典，然后按频率排序得到id)

- Don't use standard preprocessing steps like stemming or stopword removal when you have pre-trained embeddings in deeplearning methods

- Get your vocabulary as close to the embeddings as possible


### 以上持续更新...

## 优秀的*kernel*整理
- [LSTM is all you need](https://www.kaggle.com/mihaskalic/lstm-is-all-you-need-well-maybe-embeddings-also)
- [A look at different embeddings](https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings)
- [How to preprocessing when use embeddings](https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings)
```python
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)

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
    
sentences = train["question_text"].progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)
print({k: vocab[k] for k in list(vocab)[:5]})

from gensim.models import KeyedVectors
news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)

import operator 

def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x
    
oov = check_coverage(vocab,embeddings_index)

oov[:10]

'?' in embeddings_index

'&' in embeddings_index

def clean_text(x):

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x
    
train["question_text"] = train["question_text"].progress_apply(lambda x: clean_text(x))
sentences = train["question_text"].apply(lambda x: x.split())
vocab = build_vocab(sentences)

oov = check_coverage(vocab,embeddings_index)

oov[:10]

for i in range(10):
    print(embeddings_index.index2entity[i])
    
import re

def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x
    
train["question_text"] = train["question_text"].progress_apply(lambda x: clean_numbers(x))
sentences = train["question_text"].progress_apply(lambda x: x.split())
vocab = build_vocab(sentences)

oov = check_coverage(vocab,embeddings_index)
oov[:20]

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re
    
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

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)
    
train["question_text"] = train["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
sentences = train["question_text"].progress_apply(lambda x: x.split())
to_remove = ['a','to','of','and']
sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]
vocab = build_vocab(sentences)

oov = check_coverage(vocab,embeddings_index)
oov[:20]

```



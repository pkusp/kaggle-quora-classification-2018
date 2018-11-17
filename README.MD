> 说明：本项目为`kaggle`比赛内容，开始时间`2018.11.09`

# ***kaggle - Quora insincere classification***
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

#Train shape :  (1306122, 3)
#Test shape :  (56370, 2)

train columns = Index(['qid', 'question_text', 'target'], dtype='object')
train["target"].value_counts()
 0    1225312
 1      80810

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
- python闭包
```python
def ex_func(n):
	sum = n
	def ins_func():
		return sum+1
	return ins_func
f = ex_func(10)
f() # == 11

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

## ***Kernel Baseline Boosting*** 

- [How to preprocessing when using embeddings](https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings)
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
- [LSTM is all you need](https://www.kaggle.com/mihaskalic/lstm-is-all-you-need-well-maybe-embeddings-also)
```python
# bi-LSTM
```
- [A look at different embeddings](https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings)
```python
# try different embeddings
```



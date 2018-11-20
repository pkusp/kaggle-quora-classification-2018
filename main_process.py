
# preprocessing first
import pandas as pd
from tqdm import tqdm
import operator 
from gensim.models import KeyedVectors
import re

tqdm.pandas()
# read data
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)

# config
news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'


class MisSpell(object):
    def __init__(self):
        # 手写常见的拼写错误
        self.mispell_dict = {'colour':'color',
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
        self.mispellings, self.mispellings_re = self._get_mispell(self.mispell_dict)
    # 从上面可知，标点相关的通用问题解决差不多了，现在开始处理拼写的问题
    # 将字典编译成正则表达式
    def _get_mispell(self,mispell_dict):
        mispell_re = re.compile('(%s)' % '|'.join(self.mispell_dict.keys()))
        return mispell_dict, mispell_re
    # 闭包处理, 将所有的错误拼写替换为标准拼写
    def replace_typical_misspell(self,text):
        def replace(match):
            return self.mispellings[match.group(0)]
        return self.mispellings_re.sub(replace, text)


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


# 检查训练集的词表和预训练的embeddings单词的交集
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
    # # 覆盖到的词典的比例 24%
    print('Found embeddings for {:.2%} of vocab'.format(len(catch_mp) / len(vocab)))
    # # 覆盖到的总词数 78%
    print('Found embeddings for  {:.2%} of all text'.format(count_catch / (count_catch + cnt_miss)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]
    return sorted_x


def clean_text(x):
    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x


# 将数字替换为#号，按长度来替换
def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


def pre_process(train):
    # vocab为词频字典
    sentences = train["question_text"].progress_apply(lambda x: x.split()).values
    vocab = build_vocab(sentences)
    # print({k: vocab[k] for k in list(vocab)[:5]})
    # 利用KeyedVectors方法读取预训练的embeddings, embeddings_index为读取的字典->「word:vector」
    embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)
    # print(embeddings_index["hello"])
    # 预训练的embeddings覆盖率太低，很多训练集中的词没有embedding！
    print("初始的训练集与embeddings覆盖情况：")
    oov = check_coverage(vocab,embeddings_index)
    # 看看top10 out of vocab的词是哪些：
    print(oov[:10])

    # print('?' in embeddings_index) # false
    # print('&' in embeddings_index) # true

    # 在读取训练集query时加入clean_text()，以去除标点
    train["question_text"] = train["question_text"].progress_apply(lambda x: clean_text(x))
    sentences = train["question_text"].apply(lambda x: x.split())
    vocab = build_vocab(sentences)
    # 去除标点后检查训练集与embeddings的重合率：由24% 78% -> 57% 90%
    print("去除标点后：")
    oov = check_coverage(vocab,embeddings_index)
    # 再次的，检查漏掉的top10单词是哪些：（大部分都是数字！）
    print("漏掉的词top10：")
    print(oov[:10])

    # 现在检查top10的embeddings看看是否有所发现
    for i in range(10):
        print(embeddings_index.index2entity[i])
    # 在读取训练集时同时clean_number()
    train["question_text"] = train["question_text"].progress_apply(lambda x: clean_numbers(x))
    sentences = train["question_text"].progress_apply(lambda x: x.split())
    vocab = build_vocab(sentences)
    # 再次的，看看处理后的embeddings和训练集的交集：57% 90% -> 60% 90%
    print("去除数字后：")
    oov = check_coverage(vocab,embeddings_index)
    # 继续看看top20漏掉的词
    print("漏掉的词top20：")
    print(oov[:20])

    mispell = MisSpell()
    # 再次的，将replace_typical_misspell()加入到训练集的读取过程，使拼写更加标准化，并去除一些embeddings漏掉的top几的词
    train["question_text"] = train["question_text"].progress_apply(lambda x: mispell.replace_typical_misspell(x))
    sentences = train["question_text"].progress_apply(lambda x: x.split())
    to_remove = ['a','to','of','and']
    sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]
    vocab = build_vocab(sentences)
    # 再次检查embeddings和训练集的交集：60% 90% -> 60% 99% 分别为（字典遗漏和总词数遗漏）
    print("去除常见拼写错误后：")
    oov = check_coverage(vocab,embeddings_index)
    # 最后检查遗漏的top20，都是新词，没问题了
    print("漏掉的词top20：")
    print(oov[:20])
    return train


train_clean = pre_process(train_df)
test_clean = pre_process(test_df)


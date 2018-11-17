> 说明：本项目为kaggle比赛内容，开始时间2018.11.09

# kaggle - Quora insincere classification
- 比赛链接:[Quora Insincere Classification](https://www.kaggle.com/c/quora-insincere-questions-classification)
- 比赛形式:kernel only


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



### 以上持续更新...

## 一些优秀的kernel
- [LSTM is all you need](https://www.kaggle.com/mihaskalic/lstm-is-all-you-need-well-maybe-embeddings-also)
- [a look at different embeddings](https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings)
- [How to preprocessing when use embeddings](https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings)



## 一些操作技巧整理

- KeyeVectors读取预训练的词向量

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
--------------------- 
原文：https://blog.csdn.net/yangfengling1023/article/details/81705109 
```
- use tqdm to see progress bar
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

- `tokenizer`用来将query转化为序列(先遍历得到字典，然后按频率排序得到id)

- Don't use standard preprocessing steps like stemming or stopword removal when you have pre-trained embeddings

- Get your vocabulary as close to the embeddings as possible

## 操作技巧梳理
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

- 小数位数
```python

```
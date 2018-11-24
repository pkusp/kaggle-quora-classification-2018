> 说明：本项目为`kaggle`比赛内容，开始时间`2018.11.09`

# ***kaggle - Quora insincere classification***
- 比赛链接:[*Quora Insincere Classification*](https://www.kaggle.com/c/quora-insincere-questions-classification)
- 比赛形式: *kernel only*

## 目录说明
> 更新中...先做数据处理，再跑LSTM的baseline，再用CNN, 再加attention，再调超参数
- [*input*](./input)为输入数据
- [*working/main_process.py*](./main_process.py) 为数据预处理，返回 `train`和`test`清洗后的数据
- [*working/bi-lstm_basline.py*](./bi-lstm_baseline.py)为基线模型

-
-
-
-


## EDA
- 数据探测：[*EDA*](./docs/eda.md)



## 一些操作技巧梳理

- 操作技巧梳理：[*tricks*](./docs/trick.md)
- 笔记：[keras学习笔记](https://www.cnblogs.com/limitlessun/p/9296614.html#_label3)


## ***Kernel Baseline Boosting*** 

- 优秀的kernel梳理：[*kernels*](./docs/kernels.md)
    - [Pre-processing when using embeddings]
    - LSTM is all you need
    - LSTM+Attention
    - Different embeddings with attention
    - LSTM + CNN
    - [LSTM Attention](.https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043)

## *submission*情况
- LSTM baseline no tune: 0.573
- Pre-processing + LSTM Baseline no tune: 0.631
-
-


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

## 一些需要处理的问题

- 新词没有词向量
```python
     [('bitcoin', 987), ('Quorans', 858), ('cryptocurrency', 822), ('Snapchat', 807), ('btech', 632), ('Brexit', 493), (
        'cryptocurrencies', 481), ('blockchain', 474), ('behaviour', 468), ('upvotes', 432), ('programme', 402), (
      'Redmi', 379), ('realise', 371), ('defence', 364), ('KVPY', 349), ('Paytm', 334), ('grey', 299), ('mtech', 281), (
      'Btech', 262), ('bitcoins', 254)]
```

- 正常的清洗之后还有异常文字(非英文)
```python
odict_keys(
    ['w', 'h', 'a', 't', ' ', 'v', 'e', 'b', 'n', 's', 'x', 'i', 'm', 'u', 'o', 'd', 'l', 'p', 'r', '?', 'c', 'g', 'f',
     ',', 'y', 'j', '9', '8', '%', '1', '0', '2', 'k', 'q', '5', '$', '6', '.', 'z', '(', ')', "'", '-', '’', '3', '7',
     '/', '!', '"', 'é', '4', '…', '&amp;', '“', '”', '+', '\\', '=', '{', '^', '}', ';', '[', ']', '|', ':', '*',
     '&lt;', '₹', 'á', '²', 'ế', '청', '하', '¨', '‘', '√', '×', '−', '´', '\xa0', '`', 'θ', '高', '端', '大', '气', '上', '档',
     '次', '_', '½', 'π', '#', '小', '鹿', '乱', '撞', '成', '语', 'ë', 'à', 'ç', '@', 'ü', 'č', 'ć', 'ž', 'đ', '&gt;', '°',
     'द', 'े', 'श', '्', 'र', 'ो', 'ह', 'ि', 'प', 'स', 'थ', 'त', 'न', 'व', 'ा', 'ल', 'ं', '林', '彪', '€', '\u200b', '˚',
     'ö', '~', '—', '越', '人', 'च', 'म', 'क', 'ु', 'य', 'ी', 'ê', 'ă', 'ễ', '∞', '抗', '日', '神', '剧', '，', '\uf02d', '–',
     '？', 'ご', 'め', 'な', 'さ', 'い', 'す', 'み', 'ま', 'せ', 'ん', 'ó', 'è', '£', '¡', 'ś', '≤', '¿', 'λ', '魔', '法', '师', '）',
     'ğ', 'ñ', 'ř', '그', '자', '식', '멀', '쩡', '다', '인', '공', '호', '흡', '데', '혀', '밀', '어', '넣', '는', '거', '보', '니', 'ǒ',
     'ú', '️', 'ش', 'ه', 'ا', 'د', 'ة', 'ل', 'ت', 'َ', 'ع', 'م', 'ّ', 'ق', 'ِ', 'ف', 'ي', 'ب', 'ح', 'ْ', 'ث', '³', '饭',
     '可', '以', '吃', '话', '不', '讲', '∈', 'ℝ', '爾', '汝', '文', '言', '∀', '禮', 'इ', 'ब', 'छ', 'ड', '़', 'ʒ', '有', '「', '寧',
     '錯', '殺', '一', '千', '絕', '放', '過', '」', '之', '勢', '㏒', '㏑', 'ू', 'â', 'ω', 'ą', 'ō', '精', '杯', 'í', '生', '懸', '命',
     'ਨ', 'ਾ', 'ਮ', 'ੁ', '₁', '₂', 'ϵ', 'ä', 'к', 'с', 'ш', 'ɾ', '\ufeff', 'ã', '©', '\x9d', 'ū', '™', '＝', 'ù', 'ɪ',
     'ŋ', 'خ', 'ر', 'س', 'ن', 'ḵ', 'ā', 'ѕ', ...])
```

- 验证集还没用完

- SEQ_LEN还需要确定

- 模型融合还未使用

- XGB还未使用

## 参考文献

- [Multilingual Hierarchical Attention Networks for Document Classification](https://arxiv.org/abs/1707.00896v1)

- [ Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://www.paperweekly.site/papers/812)
- [Effective Approaches to Attention-based Neural Machine Translation](https://www.paperweekly.site/papers/806)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://www.paperweekly.site/papers/434)
- [Attention Is All You Need](https://www.paperweekly.site/papers/224)
- [ Weighted Transformer Network for Machine Translation](https://www.paperweekly.site/papers/2013)

- [文本分类中的Attention理解](https://blog.csdn.net/fkyyly/article/details/82501126?from=singlemessage&isappinstalled=0)

- [基于Attention机制的上下文分类算法在问答系统中的作用](https://www.jianshu.com/p/13bddd67cac3?from=singlemessage&isappinstalled=0)

- [五种Attention模型方法及应用](https://blog.csdn.net/m0epNwstYk4/article/details/81073986?from=singlemessage&isappinstalled=0)


### 以上持续更新...

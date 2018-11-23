> 说明：本项目为`kaggle`比赛内容，开始时间`2018.11.09`

# ***kaggle - Quora insincere classification***
- 比赛链接:[Quora Insincere Classification](https://www.kaggle.com/c/quora-insincere-questions-classification)
- 比赛形式: *kernel only*

## 目录说明
> 更新中...
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



## ***Kernel Baseline Boosting*** 

- 优秀的kernel梳理：[*kernels*](./docs/kernels.md)



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

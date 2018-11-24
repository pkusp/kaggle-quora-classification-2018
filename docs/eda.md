
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
![F1-Score](./docs/precision-recall.jpg)

```python
precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = (TP + TN) / (TP + FP + TN + FN)
```

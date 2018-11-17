from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
l1 = "今天要上书法课"
l2 = "今天上啥课啊"

l=[l1,l2]

tk = Tokenizer(lower=True,filters="啊")
print(l)

tk.fit_on_texts(l)

train_tk = tk.texts_to_sequences(l)
test_tk = tk.texts_to_sequences(l1)
print(train_tk)

max_len = 10
X_train = pad_sequences(train_tk,max_len)
print(X_train)
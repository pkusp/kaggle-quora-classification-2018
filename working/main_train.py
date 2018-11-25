from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
# l1 = '今天要上书法课'
# l2 = '今天上啥课啊'

l=['How did Quebec nationalists see their province as a nation in the 1960s?',
 'Do you have an adopted dog, how would you encourage people to adopt and not shop?',
 'Why does velocity affect time? Does velocity affect space geometry?',
 'How did Otto von Guericke used the Magdeburg hemispheres?',
 'Can I convert montra helicon D to a mountain bike by just changing the tyres?']

tk = Tokenizer(lower=True)
print("\ninput query:\n",l)

tk.fit_on_texts(l)

train_tk = tk.texts_to_sequences(l)
# test_tk = tk.texts_to_sequences(l1)
print("\ntext to sequence:\n",train_tk)

max_len = 100
X_train = pad_sequences(train_tk,max_len)
print("\npad sequence:\n",X_train)

print("\nword index:\n",tk.word_index)
print("\nword count:\n",tk.word_counts)


#
embedding_matrix = np.random.normal(100, 2, (5, 10))
print(embedding_matrix)




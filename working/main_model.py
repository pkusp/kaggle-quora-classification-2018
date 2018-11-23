
import time,math
from tqdm import tqdm
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Bidirectional

t_start=time.time()


class BiLstmModel(object):
    def __init__(self):
        self.model=Sequential()
        self.batch_size=128

    def get_data(self,train_df,val_x,val_y)
        self.train_df = train_df
        self.val_x = val_x
        self.val_y = val_y
        
    def train_batch_gen(self,train_df):
        batch_size=self.batch_size
        n_batches = math.ceil(len(train_df) / batch_size)  # ceil为上取整，求batch数量
        while True: 
            train_df = self.train_df.sample(frac=1.)  # Shuffle the data. 打乱训练集顺序（100%采样）
            for i in range(n_batches):  # 从训练集中顺序取每个batch数据，句子在第一列
                texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]
                text_arr = np.array([text_to_array(text) for text in texts])  # [query转化为三维张量]
                yield text_arr, np.array( train_df["target"][i*batch_size:(i+1)*batch_size] )  # 每次yield一个batch的train和对应的label

    def add_model(self):
        self.model.add(Bidirectional(CuDNNLSTM(64,return_sequences=True),
                                        input_shape=(30,300)))
        self.model.add(Bidirectional(CuDNNLSTM(64)))
        self.model.add(Dense(1,activation="sigmoid"))
    
    def compile_model(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

    def train_model(self):
        mg=self.train_batch_gen(self.train_df)
        self.model.fit_generator(mg,epochs=20,
                            steps_per_epoch=1000,
                            validation_data=(self.val_x,self.val_y),
                            verbose=True)

    def predict(self,test_df):
        all_preds=[]
        for x in tqdm(self.test_batch_gen(test_df)):
            all_preds.extend(model.predict(x).flatten())
        y_test = (np.array(all_preds)>0.5).astype(np.int)
        submit_df = pd.DataFrame({"qid":test_df["qid"],"prediction":y_test})
        submit_df.to_csv("submition.csv",index=False)


    


print("done")
t_end = time.time()
print("run {}s".format(round(t_end-t_start)))

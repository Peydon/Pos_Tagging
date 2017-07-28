from keras.models import Model,load_model
from keras.layers import LSTM,Masking,Input,Dense,Dropout,TimeDistributed
from keras.utils import to_categorical
from src.data_process import datasets
import numpy as np


# 加载数据

d = datasets()
sentences, seq_tags, MAX_SEQ_LEN,MIN_SEQ_LEN,vector_model = d.load_PFR_data()


# 输入输出维度
input_dim = 200
output_dim = 42
# 总样本数
samples_num = len(sentences)

# 训练，测试数目 9:1
train_num = int(samples_num * 9 / 10)

# 样本序列
x_samples = np.zeros(shape=(samples_num, MAX_SEQ_LEN, input_dim))
y_samples = np.zeros(shape=(samples_num, MAX_SEQ_LEN, output_dim))

for i in range(len(sentences)):
    for j in range(len(sentences[i])):
        x_samples[i][j] = sentences[i][j]
        y_samples[i][j] = to_categorical(seq_tags[i][j], output_dim)

# 划分训练和测试样本 9:1
x_train = x_samples[:train_num]
y_train = y_samples[:train_num]
x_test = x_samples[train_num + 1:]
y_test = y_samples[train_num + 1:]

def train():
    # LSTM
    hidden_unit = 200
    # dropout
    drop_out_rate = 0.25

    # 输入为一个句子的单词序列
    input_seq = Input(shape=(None, input_dim))
    # masking
    mask_out = Masking(mask_value=0.0)(input_seq)
    # LSTM
    lstm_out = LSTM(units=hidden_unit, return_sequences=True)(mask_out)
    # drop_out
    drop_out = Dropout(drop_out_rate)(lstm_out)
    # softmax
    output_seq = TimeDistributed(Dense(output_dim, activation="softmax"))(drop_out)

    # 编译
    model = Model(inputs=input_seq, outputs=output_seq)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(
        x=x_train, y=y_train, epochs=15
        , batch_size=32, validation_split=0.1, shuffle=0, verbose=2
    )
    model.save('../../../result/lstm.h5')
    print(model.evaluate(x=x_test, y=y_test, verbose=1))


def test():
    model=load_model('../../../result/lstm.h5')
    print(model.evaluate(x=x_test,y=y_test))
train()

test()

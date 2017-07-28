from keras.models import Model,load_model
from keras.optimizers import Adam
from keras.layers import Conv1D,LSTM,MaxPool1D,Input,ZeroPadding1D,Dense,Dropout,TimeDistributed
from keras.utils import to_categorical
from src.data_process import datasets
import  random
import numpy as np
# 加载数据
d = datasets()
sentences, seq_tags, MAX_SEQ_LEN ,MIN_SEQ_LEN,vertor_model= d.load_PFR_data()

# 输入输出维度
input_dim = 200
output_dim = 42

# CNN
filters_num=200
kernel_size=3
pool_size=3
strides=1
# LSTM
hidden_unit = 200
# Dense
drop_out_rate = 0.25
# 长度区间
len_nums=MAX_SEQ_LEN-MIN_SEQ_LEN+1
# 总样本数
samples_num = len(sentences)
#bacth
bacth_size=32
#epoch
step_epochs=int(samples_num/32)

#根据不同长度句子划分到不同的nd_array里
len_x_samples = [[] for i in range(len_nums)]
len_y_samples = [[] for j in range(len_nums)]
for sentence, tags in zip(sentences, seq_tags):
    len_x_samples[len(sentence) - MIN_SEQ_LEN].append(sentence)
    len_y_samples[len(sentence) - MIN_SEQ_LEN].append(to_categorical(tags, output_dim))
len_x_samples = [np.array(s) for s in len_x_samples]
len_y_samples = [np.array(s) for s in len_y_samples]

train_x,train_y,valid_x,valid_y,test_x,test_y=[],[],[],[],[],[]

#划分测试集验证集和测试集
for sample_x,sample_y in zip(len_x_samples,len_y_samples):
    l=len(sample_x)
    a=int(l*8/10)
    b=int(l*1/10)
    train_x.append(sample_x[:a])
    train_y.append(sample_y[:a])
    valid_x.append(sample_x[a+1:a+b])
    valid_y.append(sample_y[a+1:a+b])
    test_x.append(sample_x[a+b+1:])
    test_y.append(sample_y[a+b+1:])


#获得训练集
def generate_train_arrays():
    while True:
        index = list(range(len_nums))
        len_left = {i: len(train_x[i]) for i in range(len_nums)}
        random.shuffle(index)
        while(True):
            find=0
            for i in index:
                left=len_left[i]
                if left<bacth_size:
                    continue
                L = len(train_x[i])
                x=np.array(train_x[i][L-left:L-left+bacth_size])
                y=np.array(train_y[i][L-left:L-left+bacth_size])
                len_left[i]-=bacth_size
                find=1
                yield (x,y)
            if find==0:break
#获得验证集
def generate_valid_arrays():
    while 1:
        for x,y in zip(valid_x,valid_y):
            yield (x,y)
#获得测试集
def generate_test_arrays():
    while 1:
        for x,y in zip(test_x,test_y):
            yield (x,y)

def train():

    # 输入为一个句子的单词序列
    input_seq = Input(shape=(None, input_dim), )
    #卷积，RELU
    conv_out=Conv1D(filters=filters_num,kernel_size=kernel_size,padding='same',activation='relu',use_bias=1,)(input_seq)
    #在头部zero pad 2
    pad_out=ZeroPadding1D(padding=(kernel_size-1,0))(conv_out)
    #最大池化
    pool_out=MaxPool1D(pool_size=pool_size,strides=strides,padding='valid')(pad_out)
    # LSTM
    lstm_out = LSTM(units=hidden_unit, return_sequences=True,)(pool_out)
    # drop_out
    drop_out = Dropout(drop_out_rate)(lstm_out)
    # softmax
    output_seq = TimeDistributed(Dense(output_dim, activation="softmax"))(drop_out)

    # 编译
    model = Model(inputs=input_seq, outputs=output_seq)
    model.compile(optimizer='Rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # 结构
    model.summary()

    model.fit_generator(generator=generate_train_arrays(),
                        epochs=50,
                        steps_per_epoch=step_epochs,
                        validation_data=generate_valid_arrays(),
                        validation_steps=len_nums,
                        verbose=2)
    model.save('../../../result/CNN_lstm.h5')
    print(model.evaluate_generator(generator=generate_test_arrays(), steps=len_nums))

def test():
    md=load_model('../../../result/CNN_lstm.h5')
    print(md.evaluate_generator(generator=generate_test_arrays(),steps=len_nums))
train()
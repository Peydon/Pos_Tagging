from keras.models import Model,load_model
from keras.layers import Conv1D,LSTM,MaxPool1D,Input,ZeroPadding1D,Dense,Dropout,TimeDistributed
from src.data_process import datasets

# 加载数据
d = datasets()
train_x,train_y,valid_x,valid_y,test_x,test_y,samples= d.load_PFR_data()

# 输入输出维度
input_dim = 200
output_dim = 42

#bacth
bacth_size=32

# CNN
filters_num=200
kernel_size=3
pool_size=3
strides=1

# LSTM
hidden_unit = 200

# Dense
drop_out_rate = 0.5

#steps_epoch
steps_epoch=int(samples/bacth_size)


def train():

    # 输入为一个句子的单词序列
    input_seq = Input(shape=(None, input_dim),)

    #drop_out
    x = Dropout(drop_out_rate)(input_seq)

    #卷积，RELU
    x = Conv1D(filters=filters_num,kernel_size=kernel_size,padding='same',activation='relu',use_bias=1,)(x)

    #在头部zero pad 2
    x = ZeroPadding1D(padding=(kernel_size-1,0))(x)

    #drop_out
    x = Dropout(drop_out_rate)(x)

    #最大池化
    x = MaxPool1D(pool_size=pool_size,strides=strides,padding='valid')(x)

    # LSTM
    x = LSTM(units=hidden_unit, return_sequences=True,)(x)

    # drop_out
    x = Dropout(drop_out_rate)(x)

    # softmax
    output_seq = TimeDistributed(Dense(output_dim, activation="softmax"))(x)

    # 编译
    model = Model(inputs=input_seq, outputs=output_seq)
    model.compile(optimizer='Rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # 结构
    model.summary()
    model.fit_generator(generator=d.generate_train_arrays(train_x,train_y,bacth_size=bacth_size,),
                        epochs=50,
                        steps_per_epoch=steps_epoch,
                        validation_data=d.generate_valid_arrays(valid_x,valid_y),
                        validation_steps=len(valid_x),
                        verbose=2)

    #保存验证
    model.save('../../../result/CNN_lstm.h5')
    print(model.evaluate_generator(generator=d.generate_test_arrays(test_x,test_y), steps=len(test_x)))

train()

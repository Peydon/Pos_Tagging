from keras.models import Model
from keras.layers import LSTM,Input,Dense,Dropout,TimeDistributed
from src.data_process import datasets


# 加载数据
d = datasets()
train_x,train_y,valid_x,valid_y,test_x,test_y,samples= d.load_PFR_data()

# 输入输出维度
input_dim = 200
output_dim = 42

#bacth
bacth_size=32

# LSTM
hidden_unit = 200

# Dense
drop_out_rate = 0.5

#steps_epoch
steps_epoch=int(samples/bacth_size)


def train():
    # 输入为一个句子的单词序列
    input_seq = Input(shape=(None, input_dim))
    # LSTM
    lstm_out = LSTM(units=hidden_unit, return_sequences=True)(input_seq)
    # drop_out
    drop_out = Dropout(drop_out_rate)(lstm_out)
    # softmax
    output_seq = TimeDistributed(Dense(output_dim, activation="softmax"))(drop_out)

    # 编译
    model = Model(inputs=input_seq, outputs=output_seq)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(generator=d.generate_train_arrays(train_x, train_y, bacth_size=bacth_size, ),
                        epochs=20,
                        steps_per_epoch=steps_epoch,
                        validation_data=d.generate_valid_arrays(valid_x, valid_y),
                        validation_steps=len(valid_x),
                        verbose=2)

    # 保存验证
    model.save('../../../result/lstm.h5')
    print(model.evaluate_generator(generator=d.generate_test_arrays(test_x, test_y), steps=len(test_x)))

train()

from keras.models import load_model,Sequential
from keras.layers import Conv1D,LSTM,ZeroPadding1D,Dense,Dropout,TimeDistributed
from src.data_process import datasets

# 加载数据
d = datasets()
train_x,train_y,valid_x,valid_y,test_x,test_y,samples= d.load_PFR_data('199801')

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
hidden_unit = 100

# Dense
drop_out_rate = 0.5

#steps_epoch
steps_epoch=int(samples/bacth_size)


def train():
    model=Sequential(
        [
            ZeroPadding1D(padding=(1, 1),input_shape=(None,200)),
            Conv1D(filters=filters_num, kernel_size=kernel_size, padding='valid', activation='relu', use_bias=1, ),
            LSTM(units=hidden_unit, return_sequences=True, ),
            Dropout(drop_out_rate),
            TimeDistributed(Dense(output_dim, activation="softmax"))

        ]
    )
    # 编译
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

def re_train():
    model=load_model('../../../result/CNN_lstm.h5')
    model.fit_generator(generator=d.generate_train_arrays(train_x, train_y, bacth_size=bacth_size, ),
                        epochs=40,
                        steps_per_epoch=steps_epoch,
                        validation_data=d.generate_valid_arrays(valid_x, valid_y),
                        validation_steps=len(valid_x),
                        verbose=2)
    model.save('../../../result/CNN_lstm.h5')
    print(model.evaluate_generator(generator=d.generate_test_arrays(test_x, test_y), steps=len(test_x)))

def predict(nd_array):
    model=load_model('../../../result/CNN_lstm.h5')
    model.predict_classes()

from keras.models import load_model,Sequential
from keras.layers.crf import ChainCRF,create_custom_objects
from keras.layers import Conv1D,LSTM,ZeroPadding1D,Dense,Dropout,TimeDistributed,Bidirectional,BatchNormalization,Embedding
from src.data_process2 import datasets

# 加载数据
d = datasets()
embedding_matrix,train_x,train_y,valid_x,valid_y,test_x,test_y,samples= d.load_PFR_data('PFR')

# 输入输出维度
input_dim = 200
output_dim = 42

#bacth
bacth_size=256

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
    model=Sequential()
    model.add(Embedding(len(embedding_matrix),
                            input_dim,
                            weights=[embedding_matrix],
                            trainable=True))
    #model.add(BatchNormalization())
    #model.add(Dropout(drop_out_rate))
    model.add(ZeroPadding1D(padding=(1, 1)))
    model.add(Conv1D(filters=filters_num, kernel_size=kernel_size, padding='valid', activation='relu', use_bias=1, ))
    model.add(BatchNormalization())
    #model.add(Dropout(drop_out_rate))
    model.add(Bidirectional(LSTM(units=hidden_unit, return_sequences=True, recurrent_dropout=drop_out_rate),
                      input_shape=(None, input_dim), merge_mode='sum'),)
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_rate))
    model.add(TimeDistributed(Dense(output_dim)))
    model.add(BatchNormalization())
    crf=ChainCRF()
    model.add(crf)
    # 编译
    model.compile(optimizer='adam',
                  loss=crf.loss,
                  metrics=["acc"])
    # 结构
    model.summary()
    model.fit_generator(generator=d.generate_train_arrays(train_x,train_y,bacth_size=bacth_size,),
                        epochs=30,steps_per_epoch=steps_epoch,
                        validation_data=d.generate_valid_arrays(valid_x,valid_y),
                        validation_steps=len(valid_x),
                        verbose=2)

    #保存验证
    model.save('../../../result/CNN_bilstm+crf.h5')
    print(model.evaluate_generator(generator=d.generate_test_arrays(test_x,test_y), steps=len(test_x)))

def re_train():
    model=load_model('../../../result/CNN_bilstm+crf.h5',)
    model.fit_generator(generator=d.generate_train_arrays(train_x, train_y, bacth_size=bacth_size, ),
                        epochs=2,
                        steps_per_epoch=steps_epoch,
                        validation_data=d.generate_valid_arrays(valid_x, valid_y),
                        validation_steps=len(valid_x),
                        verbose=2)
    model.save('../../../result/CNN_bilstm+crf.h5')
    print(model.evaluate_generator(generator=d.generate_test_arrays(test_x, test_y), steps=len(test_x)))

def predict(nd_array):
    model=load_model('../../../result/CNN_bilstm+crf.h5')
    input(a="")

    model.predict_classes()
train()
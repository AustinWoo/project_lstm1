import pandas as pd


import numpy as np
from matplotlib import pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# batch_size = 200
# epochs = 20
filename_train = '/Users/Austin.Woo/Downloads/ADB_export_adm_raw_export_h1_lstm_v1_part1.csv'
filename_test = '/Users/Austin.Woo/Downloads/ADB_export_adm_raw_export_h1_lstm_v1_part2.csv'
filename = '/Users/Austin.Woo/Downloads/ADB_export_adm_raw_export_h1_lstm_v1.csv'

def load_dataset(filename):
    # 导入数据
    dataset = pd.read_csv(filename, index_col=0)
    return dataset


def build_model(lstm_input_shape):
    model = Sequential()
    # model.add(LSTM(units=50, input_shape=lstm_input_shape, return_sequences=True))
    # model.add(LSTM(units=50, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(units=100, input_shape=lstm_input_shape))
    model.add(Dense(1))
    #model.compile(loss='mae', optimizer='adam')
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model


if __name__ == '__main__':
    scaler = MinMaxScaler()
    #读取数据
    ds = load_dataset(filename)
    dsv = ds.values.astype('float32')

    ds_train = dsv[0:6208, :]
    ds_test = dsv[6208:6308, :]

    # print('ds_train.shape -> ', ds_train.shape)
    # print('ds_test.shape -> ', ds_test.shape)

    x_train, y_train = ds_train[:, :-1], ds_train[:, -1]
    x_test, y_test = ds_test[:, :-1], ds_test[:, -1]
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    # print('x_train.shape -> ', x_train.shape)
    # print('y_train.shape -> ', y_train.shape)
    # print('x_test.shape -> ', x_test.shape)
    # print('y_test.shape -> ', y_test.shape)

    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    # 查看数据维度
    print('x_train.shape[after] -> ', x_train.shape)
    print('x_train.shape[0] -> ', x_train.shape[0])
    print('x_train.shape[1] -> ', x_train.shape[1])
    print('x_train.shape[2] -> ', x_train.shape[2])

    # 训练模型
    lstm_input_shape = (x_train.shape[1], x_train.shape[2])
    model = build_model(lstm_input_shape)
    train_history = model.fit(x=x_train, y=y_train, batch_size=200, epochs=50, verbose=2, validation_split=0.2)
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.show()
    print('loss -> \n', train_history.history['loss'])
    print('val_loss -> \n', train_history.history['val_loss'])

    # 训练数据集评估
    scores_train = model.evaluate(x_train, y_train, batch_size=200, verbose=1)
    print('训练集评估:', model.metrics_names, ' -> ', scores_train)
    scores_test = model.evaluate(x_test, y_test, batch_size=200, verbose=1)
    print('测试集评估:', model.metrics_names, ' -> ', scores_test)

    # 使用模型预测数据
    predict_train = model.predict(x_train)
    predict_test = model.predict(x_test)
    print('y_train -> ', y_train)
    print(y_train.shape)
    print('predict_train ->', predict_train)
    print(predict_train.shape)
    print('predict_train[:, 0] -> ', predict_train[:, 0])
    print(predict_train[:, 0].shape)

    # 评估模型
    train_score = math.sqrt(mean_squared_error(y_train, predict_train[:, 0]))
    print('Train Score: RMSE -> ', train_score)
    test_score = math.sqrt(mean_squared_error(y_test, predict_test[:, 0]))
    print('Test Score: RMSE -> ', test_score)

    sum_error = 0
    for i in range(len(predict_test)):
        if i > 0:
            sum_error = sum_error + abs(predict_test[i] - y_test[i])
    MAE_test = sum_error/len(predict_test)
    print('Test Score: MAE ->', MAE_test)

    # 图表显示
    plt.plot(y_test, color='blue', label='Actual')
    plt.plot(predict_test, color='green', label='Prediction')
    plt.legend(loc='upper right')
    plt.show()



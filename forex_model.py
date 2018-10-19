
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn import metrics
import math


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


def model_score(y_actual, y_predict):
    # 评估模型
    train_score = math.sqrt(metrics.mean_squared_error(y_actual, y_predict[:, 0]))
    print('Model Score: RMSE -> ', train_score)
    sum_error = 0
    for i in range(len(y_predict)):
        if i > 0:
            sum_error = sum_error + abs(y_predict[i] - y_actual[i])
    MAE_test = sum_error/len(y_predict)
    print('Model Score: MAE ->', MAE_test)


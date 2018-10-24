
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras import optimizers
from sklearn import metrics
import math
from matplotlib import pyplot as plt


def build_model(lstm_input_shape, logfile):
    model = Sequential()
    # model.add(LSTM(units=100, input_shape=lstm_input_shape))
    # (6136, 72, 13)
    model.add(LSTM(units=90, input_shape=lstm_input_shape, return_sequences=True))
    model.add(LSTM(units=90, dropout=0.1, return_sequences=True)) #, recurrent_dropout=0.2
    model.add(LSTM(units=90, dropout=0.1))
    model.add(Dense(1))
    rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(loss='mse', optimizer=rms)

    # optimizer='rmsprop'
    # optimizer='rms'
    # optimizer='adam'

    # keras.layers.recurrent.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True,
    #                             kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    #                             bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
    #                             recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    #                             kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0,
    #                             recurrent_dropout=0.0)

    # keras.layers.core.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
    #                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
    #                         activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

    #指标 MAE	0.0004038450282180454
    #指标 MSE     3.491123079607465e-7
    #指标 RMSE   0.0005908572653024979
    model.summary()
    return model


def model_score(y_actual, y_predict, logfile):
    # 评估模型
    train_score = math.sqrt(metrics.mean_squared_error(y_actual, y_predict[:, 0]))
    print('Model Score: RMSE -> ', train_score, file=logfile)
    sum_error = 0
    for i in range(len(y_predict)):
        if i > 0:
            sum_error = sum_error + abs(y_predict[i] - y_actual[i])
    MAE_test = sum_error/len(y_predict)
    print('Model Score: MAE ->', MAE_test, file=logfile)


def model_train_plot(train_history):
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.show()
    # print('loss -> \n', train_history.history['loss'])
    # print('val_loss -> \n', train_history.history['val_loss'])

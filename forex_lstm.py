
import forex_dataset
import forex_model
import datetime
from matplotlib import pyplot as plt



class conf:
    # dataset conf
    seq_len = 72
    filename = '/Users/Austin.Woo/Downloads/ADB_export_adm_raw_export_h1_lstm_v1_part1.csv'
    fields = ['open', 'close', 'high', 'low', 'volume',	'macd_main', 'macd_signal', 'rsi', 'kdj_main', 'kdj_signal', 'ma_18', 'ma_36', 'ma_56']
    # model conf
    batch_size = 200
    epochs = 20


if __name__ == '__main__':

    t1 = datetime.datetime.now()
    print('start build dataset -> ', t1)
    ds = forex_dataset.load_dataset(conf.filename, conf.fields)
    x_train, y_train = forex_dataset.transfer_dataset(ds, conf.seq_len, conf.fields)
    t2 = datetime.datetime.now()
    print('build dataset done -> ', t2)
    print('dataset cost -> ', (t2-t1).seconds, 'seconds')
    print('x_train.shape -> ', x_train.shape)
    print('y_train.shape -> ', y_train.shape)

    # 训练模型
    lstm_input_shape = (x_train.shape[1], x_train.shape[2])
    model = forex_model.build_model(lstm_input_shape)
    t3 = datetime.datetime.now()
    print('start train model -> ', t3)
    train_history = model.fit(x=x_train, y=y_train, batch_size=conf.batch_size, epochs=conf.epochs, verbose=2, validation_split=0.2)
    t4 = datetime.datetime.now()
    print('train model done -> ', t4)
    print('train model cost ->', (t4-t3).seconds/60, 'mins')
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.show()
    # print('loss -> \n', train_history.history['loss'])
    # print('val_loss -> \n', train_history.history['val_loss'])


    # 训练数据集评估
    scores_train = model.evaluate(x_train, y_train, batch_size=200, verbose=1)
    print('训练集评估:', model.metrics_names, ' -> ', scores_train)
    # scores_test = model.evaluate(x_test, y_test, batch_size=200, verbose=1)
    # print('测试集评估:', model.metrics_names, ' -> ', scores_test)

    # 使用模型预测数据
    predict_train = model.predict(x_train)

    # 评估模型
    forex_model.model_score(y_train, predict_train)

    print('y_train -> ', y_train)
    print('predict_train -> ', predict_train)

    #
    # 图表显示训练结果
    # plt.plot(y_test, color='blue', label='Actual')
    # plt.plot(predict_test, color='green', label='Prediction')
    # plt.legend(loc='upper right')
    # plt.show()



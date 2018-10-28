
import forex_dataset
import forex_model
import datetime


class Conf:
    # dataset conf
    seq_len = 12
    filename = '/Users/Austin.Woo/Downloads/ADB_export_adm_raw_export_h1_lstm_v1_part1.csv'
    filename_test = '/Users/Austin.Woo/Downloads/ADB_export_adm_raw_export_h1_lstm_v1_part2.csv'
    # fields = ['open', 'close', 'high', 'low', 'volume',	'macd_main', 'macd_signal', 'rsi', 'kdj_main', 'kdj_signal',  'adx_main', 'adx_plusdi', 'adx_minusdi',  'ma_18', 'ma_36', 'ma_56', 'mfi', 'sar', 'cci', 'wpr', 'boll_main', 'boll_upper', 'boll_lower', 'price_4avg', 'price_2avg']
    fields = ['open', 'close', 'high', 'low']
    shift = -1
    # model conf
    batch_size = 500
    epochs = 10


if __name__ == '__main__':
    # 准备日志文件
    logfile_path = 'log/log_' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.txt'
    logfile = open(logfile_path, 'w')

    # 准备数据
    t1 = datetime.datetime.now()
    print('start build dataset -> ', t1, file=logfile)
    # 训练数据
    ds = forex_dataset.load_data(Conf.filename, Conf.fields, Conf.shift)
    x_train = forex_dataset.x_dataset(ds, Conf.seq_len, Conf.fields, logfile)
    y_train = forex_dataset.y_dataset(ds, Conf.seq_len, logfile)
    # 测试数据
    ds_test = forex_dataset.load_data(Conf.filename_test, Conf.fields, Conf.shift)
    x_test = forex_dataset.x_dataset(ds_test, Conf.seq_len, Conf.fields, logfile)
    y_test = forex_dataset.y_dataset(ds_test, Conf.seq_len, logfile)

    t2 = datetime.datetime.now()
    print('build dataset done -> ', t2, file=logfile)
    print('dataset cost -> ', (t2-t1).seconds, 'seconds', file=logfile)
    print('x_train.shape -> ', x_train.shape, file=logfile)
    print('y_train.shape -> ', y_train.shape, file=logfile)
    print('x_test.shape ->', x_test.shape, file=logfile)
    print('y_test,shape ->', y_test.shape, file=logfile)
    print('\n', file=logfile)

    # 训练模型
    lstm_input_shape = (x_train.shape[1], x_train.shape[2])
    model = forex_model.build_model(lstm_input_shape, logfile)
    t3 = datetime.datetime.now()
    print('start train model -> ', t3, file=logfile)
    train_history = model.fit(
        x=x_train, y=y_train, batch_size=Conf.batch_size, epochs=Conf.epochs, verbose=2, validation_split=0.1)
    t4 = datetime.datetime.now()
    print('train model done -> ', t4, file=logfile)
    print('train model cost ->', (t4-t3).seconds/60, 'mins', file=logfile)
    print('\n', file=logfile)

    # 显示模型训练过程
    forex_model.model_train_plot(train_history)

    # 训练数据集评估
    scores_train = model.evaluate(x_train, y_train, batch_size=Conf.batch_size, verbose=1)
    print('训练集评估:', model.metrics_names, ' -> ', scores_train, file=logfile)
    score_test = model.evaluate(x_test, y_train, batch_size=Conf.batch_size, verbose=1)
    print('测试集评估:', model.metrics_names, ' -> ', score_test, file=logfile)

    # 使用模型预测数据
    predict_train = model.predict(x_train)
    predict_test = model.predict(x_test)

    # 评估模型
    print('=====训练集分数=====', file=logfile)
    forex_model.model_score(y_train, predict_train, logfile=logfile)
    print('=====测试集分数=====', file=logfile)
    forex_model.model_score(x_test, predict_test, logfile=logfile)

    print('y_train -> ', y_train, file=logfile)
    print('predict_train -> ', predict_train, file=logfile)

    #
    # 图表显示训练结果
    # plt.plot(y_test, color='blue', label='Actual')
    # plt.plot(predict_test, color='green', label='Prediction')
    # plt.legend(loc='upper right')
    # plt.show()

    logfile.close()

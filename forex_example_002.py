import numpy as np
import pandas as pd
import forex_dataset
import datetime


class Conf:
    # dataset conf
    seq_len = 3
    filename = '/Users/Austin.Woo/Downloads/ADB_export_adm_raw_export_h1_lstm_v1_part1.csv'
    filename_test = '/Users/Austin.Woo/Downloads/ADB_export_adm_raw_export_h1_lstm_v1_part1.csv'
    fields = ['open', 'close', 'high', 'low', 'volume']
    # model conf
    batch_size = 500
    epochs = 30


if __name__ == '__main__':
    # 准备日志文件
    logfile_path = 'log/log_' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.txt'
    logfile = open(logfile_path, 'w')

    print('main function')
    ds = forex_dataset.load_data(Conf.filename, Conf.fields)
    print('forex_dataset.load_data -> ', ds.shape)

    x_data = forex_dataset.x_dataset(ds, Conf.seq_len, Conf.fields, logfile)
    print('forex_dataset.x_dataset -> ', x_data.shape)
    # print('forex_dataset.x_dataset -> ', x_data)

    x_data2, maxmin = forex_dataset.x_dataset_by_MinMaxScaler(ds, Conf.seq_len, Conf.fields, logfile)
    print('forex_dataset.x_dataset_by_MinMaxScaler -> ', x_data2.shape)
    # print('forex_dataset.x_dataset_by_MinMaxScaler -> ', x_data2)

    x_data3 = forex_dataset.x_dataset_by_scale(ds, Conf.seq_len, Conf.fields, logfile)
    print('forex_dataset.x_dataset_by_scale -> ', x_data3.shape)

    y_data = forex_dataset.y_dataset(ds, Conf.seq_len, -3, logfile)
    print('forex_dataset.y_dataset -> ', y_data.shape)
    print('forex_dataset.y_dataset -> ', y_data)

    y_data2 = forex_dataset.y_dataset_by_ReturnRate(ds, Conf.seq_len, -3, logfile)
    print('forex_dataset.y_dataset_by_ReturnRate -> ', y_data2.shape)
    print('forex_dataset.y_dataset_by_ReturnRate -> ', y_data2)

    logfile.close()



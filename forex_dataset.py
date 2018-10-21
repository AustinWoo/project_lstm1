from sklearn import preprocessing
import numpy as np
import pandas as pd


# class conf:
    # filename = '/Users/Austin.Woo/Downloads/ADB_export_adm_raw_export_h1_lstm_v1_part2.csv'
    # fields = ['open', 'close', 'high', 'low']


def load_dataset(filename, fields):
    # 导入数据
    ds = pd.read_csv(filename, index_col=0)
    ds.sort_index(inplace=True)
    ds = ds.ix[:, fields]
    ds['lable'] = (ds['close'].shift(-1)+ds['open'].shift(-1))/2
    # ds['lable1'] = ds['close'].shift(-1)
    # ds['lable2'] = ds['open'].shift(-1)
    ds.dropna(inplace=True)
    # ds['lable_1'] = ds['lable_1'].apply(lambda x: np.where(x >= 0.2, 0.2, np.where(x > -0.2, x, -0.2)))
    ds.reset_index(drop=True, inplace=True)
    return ds


def transfer_dataset(dataset, seq_len, fields, logfile):
    x_list = []
    y_list = []
    x_ds = dataset[fields]
    # minmax = preprocessing.MinMaxScaler()
    # x_ds = minmax.fit_transform(x_ds)
    # print('dataset transfer by MinMaxScaler', file=logfile)

    for i in range(seq_len - 1, len(dataset)):
        # a = preprocessing.scale(x_ds[i+1-seq_len: i+1])         #做scale 零均值单位方差 标准化
        # print('dataset transfer by scale')
        a = np.array(x_ds[i+1-seq_len: i+1])                  #不做标准化
        x_list.append(a)
        # print('i -> ', i)
        # print('a -> ', a)
        # print('a.mean(axis=0) -> ', a.mean(axis=0))
        # print('a.std(axis=0) -> ', a.std(axis=0))
        r = dataset['lable'][i]
        y_list.append(r)
    return np.array(x_list), np.array(y_list)


# if __name__ == '__main__':
    # ds = load_dataset(conf.filename)
    # print(ds)
    # print(ds.shape)
    # x_train, y_train = transfer_dataset(ds, 3)
    # x_train = transfer_dataset(ds)
    # print(x_train)
    # print(x_train.shape)
    # print(y_train)
    # print(y_train.shape)


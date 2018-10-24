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
    
    # 数据标准化一 data_train1 = (data_train-data_mean)/5  还原 rt=r*5+data_mean[120:140].as_matrix()

    # 数据标准化二 scaler = MinMaxScaler() data_all = scaler.fit_transform(data_all)  还原  test_y = scaler.inverse_transform(test_y)

    # 数据标准化三    归一化
    # scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # scaled = scaler.fit_transform(values)
    # MinMaxScaler 原理是啥？

    # 数据标准化四
    # 因此为了应对这种情况，我们需要使训练 / 测试数据的每个n大小的窗口进行标准化，以反映从该窗口开始的百分比变化（因此点i = 0
    # 处的数据将始终为0）。我们将使用以下方程式进行归一化，然后在预测过程结束时进行反标准化，以得到预测中的真实世界数：
    # def normalise_windows(window_data):
    #     normalised_data = []
    #     for window in window_data:
    #         normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
    #         normalised_data.append(normalised_window)
    #     return normalised_data

    # 数据标准化五
    # def normalise_windows(window_data):
    #     #数据规范化
    #     normalised_data = []
    #     for window in window_data:
    #         normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
    #         normalised_data.append(normalised_window)
    #     return normalised_data
    #
    # def denormalise_windows(normdata,data,seq_len):
    #     #数据反规范化
    #     denormalised_data = []
    #     wholelen=0
    #     for i, rowdata in enumerate(normdata):
    #         denormalise=list()
    #         if isinstance(rowdata,float)|isinstance(rowdata,np.float32):
    #             denormalise = [(rowdata+1)*float(data[wholelen][0])]
    #             denormalised_data.append(denormalise)
    #             wholelen=wholelen+1
    #         else:
    #             for j in range(len(rowdata)):
    #                 denormalise.append((float(rowdata[j])+1)*float(data[wholelen][0]))
    #                 wholelen=wholelen+1
    #             denormalised_data.append(denormalise)
    #     return denormalised_data

    # 数据标准化六
    # from sklearn.preprocessing import scale
    # for i in range(conf.seq_len-1, len(traindata)):
    #     a = scale(scaledata[i+1-conf.seq_len:i+1])
    #     train_input.append(a)
    #     c = data['return'][i]
    #     train_output.append(c)

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


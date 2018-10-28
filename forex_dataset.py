from sklearn import preprocessing
import numpy as np
import pandas as pd


# class Conf:
#     filename = 'abc.csv'
#     fields = ['']

def load_data(filename, fields, shift):
    # 导入数据
    ds = pd.read_csv(filename, index_col=0)
    ds.sort_index(inplace=True)
    ds = ds.ix[:, fields]
    ds['lable'] = (ds['close'].shift(shift)+ds['open'].shift(shift))/2
    ds.dropna(inplace=True)
    ds.reset_index(drop=True, inplace=True)
    return ds


def load_data_with_ReturnRate(filename, fields, shift):
    # 导入数据
    ds = pd.read_csv(filename, index_col=0)
    ds.sort_index(inplace=True)
    ds = ds.ix[:, fields]
    ds['lable'] = (ds['close'].shift(shift) + ds['open'].shift(shift)) / (ds['close'] + ds['open'])
    ds.dropna(inplace=True)
    ds.reset_index(drop=True, inplace=True)
    return ds


def x_dataset(dataset, seq_len, fields, logfile):
    x_list = []
    x_ds = dataset[fields]
    print('x dataset transfer by nothing', file=logfile)
    for i in range(seq_len - 1, len(x_ds)):
        a = np.array(x_ds[i + 1 - seq_len: i + 1])  # 不做标准化
        x_list.append(a)
    return np.array(x_list)


    # 数据标准化一 data_train1 = (data_train-data_mean)/5  还原 rt=r*5+data_mean[120:140].as_matrix()
def x_dataset_by_Mean(dataset, seq_len, fields, logfile):
    x_list = []
    return x_list


    # 数据标准化 by MinMaxScaler
def x_dataset_by_MinMaxScaler(dataset, seq_len, fields, logfile):
    x_list = []
    x_ds = dataset[fields]
    minmax = preprocessing.MinMaxScaler()
    x_ds = minmax.fit_transform(x_ds)
    print('x dataset transfer by MinMaxScaler', file=logfile)
    for i in range(seq_len - 1, len(x_ds)):
        a = np.array(x_ds[i + 1 - seq_len: i + 1])
        x_list.append(a)
    return np.array(x_list), minmax


    # 数据标准化 by scale
def x_dataset_by_scale(dataset, seq_len, fields, logfile):
    x_list = []
    x_ds = dataset[fields]
    print('x dataset transfer by scale', file=logfile)
    for i in range(seq_len - 1, len(x_ds)):
        a = preprocessing.scale(x_ds[i+1-seq_len: i+1])         #做scale 零均值单位方差 标准化
        x_list.append(a)
    return np.array(x_list)


# 数据标准化四
    # 因此为了应对这种情况，我们需要使训练 / 测试数据的每个n大小的窗口进行标准化，以反映从该窗口开始的百分比变化（因此点i = 0
    # 处的数据将始终为0）。我们将使用以下方程式进行归一化，然后在预测过程结束时进行反标准化，以得到预测中的真实世界数：
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
def x_dataset_by_StartByZero(dataset, seq_len, fields, logfile):
    x_list = []
    return x_list


def y_dataset(dataset, seq_len, logfile):
    y_list = []
    print('y dataset transfer by nothing', file=logfile)
    ds = dataset
    for i in range(seq_len - 1, len(ds)):
        r = ds['lable'][i]
        y_list.append(r)
    return np.array(y_list)


if __name__ == '__main__':
    print('main function')


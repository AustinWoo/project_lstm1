import pandas as pd
import numpy as np

# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from keras.optimizers import SGD
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from matplotlib import pyplot as plt
# from datetime import datetime


filename = '/Users/Austin.Woo/Downloads/ADB_export_adm_raw_export_h1_lstm_v1_part1.csv'


def load_dataset():
    # 导入数据
    dataset = pd.read_csv(filename, index_col=0)
    return dataset


if __name__ == '__main__':
    data = [["2017-10-18", 11.53, 11.69, 11.70, 11.51, 871365.0, 000001],
            ["2017-10-19", 11.64, 11.63, 11.72, 11.57, 722764.0, 000001],
            ["2017-10-20", 11.59, 11.48, 11.59, 11.41, 461808.0, 000001],
            ["2017-10-23", 11.39, 11.19, 11.40, 11.15, 1074465.0, 000001]]
    series = pd.Series(data, index=['a', 'b', 'c', 'd'])
    print(series)


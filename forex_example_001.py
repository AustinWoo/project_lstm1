import numpy as np
import pandas as pd
from sklearn import preprocessing


class Conf:
    filename = '/Users/Austin.Woo/Downloads/abc.csv'


if __name__ == '__main__':
    ds = pd.read_csv(conf.filename)
    print(ds)
    # print('\n')
    # minmax = preprocessing.MinMaxScaler()
    # ds_mx = minmax.fit_transform(ds)
    # print(ds_mx)
    # print('\n')
    # # print(ds_mx[:, 1:2])
    # # ds_mx_tiny = ds_mx[:, 1:2]
    # # print('\n')
    # ds_recover = minmax.inverse_transform(ds_mx)
    # print(ds_recover)

    ds_sca = preprocessing.scale(ds)
    print(ds_sca)
    print(ds_sca.mean(axis=0))
    print(ds_sca.std(axis=0))
    ds_sca_recover = preprocessing
    # print(ds_sca.mean())
    # print(ds_sca.std())

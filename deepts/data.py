import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch.autograd import Variable

from deepts.layers import static_embedding, dynamic_feature_cat_embedding
from deepts.utils import set_logging

from IPython import embed

logging = set_logging()


class Data(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        input_i = [Variable(self.x[0][index]), Variable(self.x[1][index]), Variable(self.x[2][index])]
        label_i = Variable(self.y[index])
        return input_i, label_i

    def __len__(self):
        return len(self.y)

class TSDataset:
    def __init__(self, df, target, static_feat_col=None, dynamic_feat_cat_dict=None, 
                dynamic_feat_real_col=None, n_back=1, n_fore=1, lag=0, sliding_window_dis=1, 
                norm=True, pkl_path=None):
        self.df = df
        self.target = target
        self.static_feat_col = static_feat_col
        self.dynamic_feat_cat_dict = dynamic_feat_cat_dict
        self.dynamic_feat_real_col = dynamic_feat_real_col
        self.n_back = n_back
        self.n_fore = n_fore
        self.lag = lag
        self.sliding_window_dis = sliding_window_dis
        self.norm = norm
        self.pkl_path = pkl_path

        self.static_val_dict = None
        self.static_feat_num_dict = None
        self.dynamic_feature_cat = None
        self.dynamic_feature_real = None
        self.time_series = None
        self.lag_feature = None
        self.scaler = None
        self.is_cached = False
        self.check()

    def check(self):
        all_columns = list(self.dynamic_feat_cat_dict.keys()) + self.dynamic_feat_real_col
        assert sum(self.df.columns.isin(all_columns)) == len(all_columns),\
            "df.columns: {}, but input columns {}".format(self.df.columns, all_columns)
        self.static_val_dict = {k:i for i, k in enumerate(self.df[self.static_feat_col].unique())}
        self.df[self.static_feat_col] = self.df[self.static_feat_col].apply(lambda x: self.static_val_dict[x])
        if self.norm:
            self.df[self.target] = self.min_max_normalize(self.df[self.target].values)
        if os.path.exists(self.pkl_path):
            self.is_cached = True
            self.load_pkl()
        logging.info("Checked Dataset.")

    def get_static_feat_num_dict(self):
        self.static_feat_ori = self.df[self.static_feat_col]
        window_size = self.n_back + self.n_fore
        static_feat_num_dict = {}
        for static_feat in self.static_feat_ori:
            static_feat_num_dict[static_feat] = \
                [(self.df[self.df[self.static_feat_col] == static_feat].shape[0] \
                    - (window_size-self.sliding_window_dis)) // self.sliding_window_dis,\
                        self.df[self.df[self.static_feat_col] == static_feat].shape[0]]
        self.static_feat_num_dict = static_feat_num_dict
        return static_feat_num_dict

    def get_dynamic_feature_cat(self, period='all'):
        dynamic_feature_cat = []
        static_feat_num_dict = self.get_static_feat_num_dict()
        for static_feat, [num, tail_idx] in static_feat_num_dict.items():
            static_feat_i = self.df[self.df[self.static_feat_col] == static_feat]\
                [list(self.dynamic_feat_cat_dict.keys())].values
            for i in range(num):
                if period == 'future':
                    dynamic_feature_cat.append(
                        static_feat_i[tail_idx-(i+1)*self.n_fore: tail_idx-i*self.n_fore]
                    )
                elif period == 'backward':
                    dynamic_feature_cat.append(
                        static_feat_i[tail_idx-self.n_back-(i+1)*self.n_fore: tail_idx-(i+1)*self.n_fore]
                    )
                elif period == 'all':
                    dynamic_feature_cat.append(
                        static_feat_i[tail_idx-self.n_back-self.n_fore-i*self.n_fore: tail_idx-i*self.n_fore]
                    )
        self.dynamic_feature_cat = np.stack(dynamic_feature_cat, axis=0)
        return torch.LongTensor(self.dynamic_feature_cat)

    def get_dynamic_feature_real(self, period='future'):
        dynamic_feature_real = []
        static_feat_num_dict = self.get_static_feat_num_dict()
        for static_feat, [num, tail_idx] in static_feat_num_dict.items():
            static_feat_i_df = self.df[self.df[self.static_feat_col] == static_feat] \
                [self.dynamic_feat_real_col]
            static_feat_i = pd.concat(
                [static_feat_i_df, self.get_lag_features(self.df[self.target], self.lag)], axis=1).values
            for i in range(num):
                if period == 'future':
                    dynamic_feature_real.append(
                        static_feat_i[tail_idx-(i+1)*self.n_fore: tail_idx-i*self.n_fore, :]
                    )
                elif period == 'backward':
                    dynamic_feature_real.append(
                        static_feat_i[tail_idx-self.n_back-(i+1)*self.n_fore: tail_idx-(i+1)*self.n_fore, :]
                    )
                elif period == 'all':
                    dynamic_feature_real.append(
                        static_feat_i[tail_idx-self.n_back-self.n_fore-i*self.n_fore: tail_idx-i*self.n_fore, :]
                    )
        self.dynamic_feature_real = np.stack(dynamic_feature_real, axis=0)
        return torch.FloatTensor(self.dynamic_feature_real)

    def get_time_series(self, period='future'):
        time_series = []
        static_feat_num_dict = self.get_static_feat_num_dict()
        for static_feat, [num, tail_idx] in static_feat_num_dict.items():
            static_feat_i = self.df[self.df[self.static_feat_col] == static_feat] \
                [self.target].values
            for i in range(num):
                if period == 'future':
                    time_series.append(
                        static_feat_i[tail_idx-(i+1)*self.n_fore: tail_idx-i*self.n_fore]
                    )
                elif period == 'backward':
                    time_series.append(
                        static_feat_i[tail_idx-self.n_back-(i+1)*self.n_fore: tail_idx-(i+1)*self.n_fore]
                    )
                elif period == 'all':
                    time_series.append(
                        static_feat_i[tail_idx-self.n_back-self.n_fore-i*self.n_fore: tail_idx-i*self.n_fore]
                    )
        self.time_series = torch.FloatTensor(np.stack(time_series, axis=0))
        return self.time_series

    def get_lag_features(self, ts, lag):
        columns=['lag_'+str(i) for i in range(lag)]
        lag_feature = pd.DataFrame(columns=columns)
        for i in range(lag):
            lag_feature['lag_'+str(i)] = ts.shift(i, fill_value=0.0)
        lag_feature.shift(1, fill_value=0.0)
        self.lag_feature = lag_feature
        return lag_feature

    def min_max_normalize(self, values):
        assert type(values) == np.ndarray, "Got values of type {}, expected type \
            `numpy.ndarray`".format(type(values))
        shape_ori = values.shape
        values = values.reshape(-1, 1)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler = scaler.fit(values)
        values_normalized = self.scaler.transform(values)
        logging.info("Y's data_min: {}, data_max: {}".format(
            self.scaler.data_min_, self.scaler.data_max_))
        return values_normalized.reshape(shape_ori)

    def save_pkl(self):
        data_dict = {
            'static_feat_num_dict': self.static_feat_num_dict,
            'dynamic_feature_cat': self.dynamic_feature_cat,
            'dynamic_feature_real': self.dynamic_feature_real,
            'time_series': self.time_series,
            'lag_feature': self.lag_feature
        }
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(data_dict, f)
        logging.info("Saved data!")

    def load_pkl(self):
        with open(self.pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
        self.static_feat_num_dict = data_dict['static_feat_num_dict']
        self.dynamic_feature_cat = data_dict['dynamic_feature_cat']
        self.dynamic_feature_real = data_dict['dynamic_feature_real']
        self.time_series = data_dict['time_series']
        self.lag_feature = data_dict['lag_feature']
        logging.info("Load data!")


def dataset_split(x, ratio_list=[6, 2, 2]):
    ratio_list = np.array([0.0] + ratio_list) / sum(ratio_list)
    index_list = list(map(int, np.cumsum(ratio_list) * len(x)))
    train_slice, valid_slice, test_slice = [slice(index_list[i], index_list[i+1]) \
                                            for i in range(len(index_list)-1)]
    return x[train_slice], x[valid_slice], x[test_slice]

def get_dataloader(dataset, dynamic_feat_cat_dict, n_back, n_fore, static_feat_dim, dyn_feat_cat_kw='future'):
    if not dataset.is_cached:
        dynamic_feature_cat = dataset.get_dynamic_feature_cat(period=dyn_feat_cat_kw)
        static_feat_num_dict = dataset.get_static_feat_num_dict()
        time_series = dataset.get_time_series(period='all')
        dataset.save_pkl()
    else:
        dataset.load_pkl()
        dynamic_feature_cat = dataset.dynamic_feature_cat
        static_feat_num_dict = dataset.static_feat_num_dict
        time_series = dataset.time_series
    static_feat = static_embedding(static_feat_num_dict, n_back, n_fore, 'all', static_feat_dim)
    dynamic_feature_cat_embed = dynamic_feature_cat_embedding(dynamic_feature_cat, dynamic_feat_cat_dict)
    ts_back, ts_fore = time_series[:, :n_back], time_series[:, n_back:]
    ts_back_train, ts_back_valid, ts_back_test = dataset_split(ts_back)
    static_feat_train, static_feat_valid, static_feat_test = dataset_split(static_feat)
    dynamic_feature_cat_embed_train, dynamic_feature_cat_embed_valid, dynamic_feature_cat_embed_test\
        = dataset_split(dynamic_feature_cat_embed)
    ts_fore_train, ts_fore_valid, ts_fore_test = dataset_split(ts_fore)

    x_train = [ts_back_train, static_feat_train, dynamic_feature_cat_embed_train]
    y_train = ts_fore_train

    x_valid = [ts_back_valid, static_feat_valid, dynamic_feature_cat_embed_valid]
    y_valid = ts_fore_valid
    
    x_test = [ts_back_test, static_feat_test, dynamic_feature_cat_embed_test]
    y_test = ts_fore_test

    return x_train, y_train, x_valid, y_valid, x_test, y_test
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from deepts.feature_column import (SparseFeat, DenseFeat, get_embedding_features, get_dense_features)
from deepts.utils import set_logging, numpy_input_fn

from IPython import embed

logging = set_logging()

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

    def get_dynamic_feature_cat(self, period='future'):
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
        return self.dynamic_feature_cat

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
        return self.dynamic_feature_real

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
        self.time_series = tf.convert_to_tensor(np.stack(time_series, axis=0), dtype=tf.float32)
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
    
    


# n_fore = input_dynamic_emb.shape[1]
# store_embed = self.store_embedding(input_cov[:,:,0])
# embed_concat = tf.concat(
#         [store_embed,
#         self.nYear_embedding(input_cov[:,:,2]),
#         self.nMonth_embedding(input_cov[:,:,3]),
#         self.mDay_embedding(input_cov[:,:,4]),
#         self.wday_embedding(input_cov[:,:,5]),
#         self.nHour_embedding(input_cov[:,:,6])],
#         axis=2)
# input_store = tf.tile(store_embed[:,0:1,:], [1, 168, 1])    # input_store: [batch_size, 168, 10]


class DeeptsData:
    def __init__(self, X, Y, feature_columns, n_back=1, n_fore=1, lag=0):
        self.X = X
        self.Y = Y
        self.feature_columns = feature_columns
        self.n_back = n_back
        self.n_fore = n_fore
        self.lag = lag
        self.scaler = None
    
    def get_raw_data(self):
        return self.X, self.Y
    
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
    
    def min_max_inverse_normalize(self, values):
        assert type(values) == np.ndarray, "Got values of type {}, expected type \
            `numpy.ndarray`".format(type(values))
        # assert not self.scaler, "Haven't define scaler!"
        shape_ori = values.shape
        values = values.reshape(-1, 1)
        values_inverse = self.scaler.inverse_transform(values)
        return values_inverse.reshape(shape_ori)

    def get_sequence_features(self, Y, lag):
        sequence_feature = pd.DataFrame()
        for i in range(lag):
            sequence_feature[str(i)] = Y.shift(i, fill_value=0.0)
        sequence_feature.shift(1, fill_value=0.0)
        return tf.convert_to_tensor(sequence_feature, dtype='float32')

    def get_deepts_data(self, norm_type='minmax', format='tensor'):
        """Format input features and labels.

        Args:   
            - **X**: pd.DataFrame.
            - **Y**: pd.Series.
        """
        if norm_type == 'minmax':
            self.Y = pd.Series(self.min_max_normalize(self.Y.values))
        features_dict = {}
        sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), self.feature_columns))
        dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), self.feature_columns))
        sparse_feature_dict = get_embedding_features(self.X, sparse_feature_columns)
        dense_feature_dict = get_dense_features(self.X, dense_feature_columns)
        for sfc in sparse_feature_columns:
            features_dict[sfc.name] = sparse_feature_dict[sfc.name]
        for dfc in dense_feature_columns:
            features_dict[dfc.name] = dense_feature_dict[dfc.name]
        if self.lag > 0:
            sequence_feature = self.get_sequence_features(self.Y, self.lag)
            features_dict['seq_feat_lag_'+str(self.lag)] = sequence_feature
        feature_ts = tf.concat(list(features_dict.values()), axis=1)
        target_ts = tf.convert_to_tensor(self.Y.values, dtype='float32')

        # format features and labels
        features, labels = [], []
        num_samples = len(target_ts) - (self.n_back + self.n_fore) + 1
        for i in range(num_samples):
            features.append(feature_ts[i: i + self.n_back, :])
            labels.append(target_ts[i + self.n_back: i + self.n_back + self.n_fore])
        if format == 'tensor':
            features = tf.convert_to_tensor(features, dtype=tf.float32)
            labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        elif format == 'numpy':
            features = np.array(features)
            labels = np.array(labels)
        elif format == 'python':
            features = features
            labels = labels

        return features, labels
    
    def get_deepts_estimator_data(self, X, Y, feature_columns, window_size, n_back, n_fore, lag, batch_size):
        x, y = self.get_deepts_data(format='numpy')
        input_fn = numpy_input_fn(x, y, shuffle=False)

        return input_fn

def dataset_split(x, ratio_list=[6, 2, 2]):
    ratio_list = np.array([0.0] + ratio_list) / sum(ratio_list)
    index_list = list(map(int, np.cumsum(ratio_list) * len(x)))
    train_slice, valid_slice, test_slice = [slice(index_list[i], index_list[i+1]) \
                                            for i in range(len(index_list)-1)]
    return x[train_slice], x[valid_slice], x[test_slice]
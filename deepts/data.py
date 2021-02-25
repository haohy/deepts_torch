import os
import math
import numpy as np
import pandas as pd

import tensorflow as tf

from deepts.feature_column import (SparseFeat, DenseFeat, get_embedding_features, get_dense_features)
from deepts.utils import set_logging, numpy_input_fn

logging = set_logging()

class Dataset:
    """Base class for `Dataset`.

    Args:
        df: pandas.DataFrame, columns can't be None.
        emb_dict: Dictionary, {'day_of_week': 3} represent the column 'day_of_week' is needed to be 
            embedded and the embedding dimensionality is `3`, the dimensionality `0` represent no 
            need of embedding, i.e., using one-hot for that column.
        name: String, name of the dataset.
        
    """
    def __init__(self, df, emb_dict, name):
        self.df = df
        self.emb_dict = emb_dict
        self._name = name
        self._feature_columns = None
    
    def get_raw_data(self):
        return self.df

    @property
    def name(self):
        return self._name


class TSDataset(Dataset):
    def __init__(self, feat_dict, n_back, n_fore, lag, sliding_window_dis, name):
        self._feat_dict = feat_dict
        self.n_back = n_back
        self.n_fore = n_fore
        self.lag = lag
        self.sliding_window_dis = sliding_window_dis
        self._dynamic_feature = None
        self._static_dict = None
        self._time_series = None
        self._name = name

    def check_df(self):
        if self._static_feature is None or self._dynamic_feature in None:
            raise ValueError("TSDataset doesn't has dataframe.")
        if not (self._dynamic_feature.columns == list(self.feat_dict.keys())):
            raise ValueError("Dynamic columns {} should be same with it's feat_dict {}"
                            .format(self.df_raw.columns, list(self._feat_dict.keys())))

    def get_time_series(self, df):
        self._time_series = df
        self._static_dict = {col: i for i, col in enumerate(df.columns)}

    def get_features(self, df_feat):
        self._dynamic_feature = df_feat

    def from_csv(self, file_path, csv_kwargs):
        df_csv = pd.read_csv(file_path, **csv_kwargs)
        self.df_raw = df_csv

    def df_to_dataset(self, n_back, n_fore, lag):
        self.check_df()
        static_cols = [k for k, v in self._feat_dict.items() if v[0] == 0]
        dynamic_cols = [k for k, v in self._feat_dict.items() if v[0] == 1]
        
        
    
    def get_sparse_feat_cols(self, feat_col_dict):
        feat_cols = []
        for key, val in feat_col_dict:
            feat_cols.append(SparseFeat(key, val[0], val[2], val[3], val[4], 'int64'))
        return feat_cols

    def get_dense_feat_cols(self, feat_col_dict):
        feat_cols = []
        for key, val in feat_col_dict:
            feat_cols.append(DenseFeat(key, val[0], val[2], val[4], 'float32'))
        return feat_cols

    def get_features(self):
        sparse_feat_cols, dense_feat_cols = [], []
        sparse_feat_dict = {k: v for k, v in self._feat_cols.items() if v[1] == 1}
        dense_feat_dict = {k: v for k, v in self._feat_cols.items() if v[1] == 0}
        sparse_feat_cols = self.get_sparse_feat_cols(sparse_feat_dict)
        dense_feat_cols = self.get_dense_feat_cols(dense_feat_dict)

    @property
    def name(self):
        return self._name

    @property
    def feat_dict(self):
        return self._feat_dict

    @property
    def target_col(self):
        return self._target_col


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


class DeeptsData(Dataset):
    def __init__(self, X, Y, feature_columns, name, n_back=1, n_fore=1, lag=0):
        self.X = X
        self.Y = Y
        self.feature_columns = feature_columns
        self.n_back = n_back
        self.n_fore = n_fore
        self.lag = lag
        self.scaler = None
        super(DeeptsData, self).__init__(name)
    
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

def dataset_split(x, y, ratio_list=[6, 2, 2]):
    ratio_list = np.array([0.0] + ratio_list) / sum(ratio_list)
    index_list = list(map(int, np.cumsum(ratio_list) * len(y)))
    train_slice, valid_slice, test_slice = [slice(index_list[i], index_list[i+1]) \
                                            for i in range(len(index_list)-1)]
    return x[train_slice], y[train_slice],\
        x[valid_slice], y[valid_slice],\
        x[test_slice], y[test_slice]
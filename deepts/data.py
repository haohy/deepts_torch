import os
import math
import numpy as np
import pandas as pd

import tensorflow as tf

from deepts.feature_column import (SparseFeat, DenseFeat, get_embedding_features, get_dense_features)
from deepts.utils import set_logging, numpy_input_fn

logging = set_logging()

class Dataset:
    def __init__(self, name):
        self._name = name
    
    def get_raw_data(self):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

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
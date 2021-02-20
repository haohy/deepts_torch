import os
import math
import numpy as np
import pandas as pd

import tensorflow as tf

from deepts.feature_column import (SparseFeat, DenseFeat, get_embedding_features, get_dense_features)
from deepts.utils import numpy_input_fn


def get_sequence_features(Y, lag):
    sequence_feature = pd.DataFrame()
    for i in range(lag):
        sequence_feature[str(i)] = Y.shift(i, fill_value=0.0)
    sequence_feature.shift(1, fill_value=0.0)
    return tf.convert_to_tensor(sequence_feature, dtype='float32')

def get_deepts_data(X, Y, feature_columns, n_back=1, n_fore=1, lag=0, format='tensor'):
    """Format input features and labels.

    Args:   
        - **X**: pd.DataFrame.
        - **Y**: pd.Series.
    """
    features_dict = {}
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
    sparse_feature_dict = get_embedding_features(X, sparse_feature_columns)
    dense_feature_dict = get_dense_features(X, dense_feature_columns)
    for sfc in sparse_feature_columns:
        features_dict[sfc.name] = sparse_feature_dict[sfc.name]
    for dfc in dense_feature_columns:
        features_dict[dfc.name] = dense_feature_dict[dfc.name]
    if lag > 0:
        sequence_feature = get_sequence_features(Y, lag)
        features_dict['seq_feat_lag_'+str(lag)] = sequence_feature
    feature_ts = tf.concat(list(features_dict.values()), axis=1)
    target_ts = tf.convert_to_tensor(Y.values, dtype='float32')

    # format features and labels
    features, labels = [], []
    num_samples = len(target_ts) - (n_back + n_fore) + 1
    for i in range(num_samples):
        features.append(feature_ts[i: i + n_back, :])
        labels.append(target_ts[i + n_back: i + n_back + n_fore])
    if format == 'tensor':
        features = tf.convert_to_tensor(features)
        labels = tf.convert_to_tensor(labels)
    elif format == 'numpy':
        features = np.array(features)
        labels = np.array(labels)
    elif format == 'python':
        features = features
        labels = labels

    return features, labels
    
def get_deepts_estimator_data(X, Y, feature_columns, window_size, n_back, n_fore, lag, batch_size):
    x, y = get_deepts_data(X, Y, feature_columns, window_size, n_back, n_fore, lag, format='numpy')

    # def input_fn():
    #     dataset = tf.data.Dataset.from_tensor_slices((x, y))
    #     dataset = dataset.batch(batch_size)
    #     return dataset

    input_fn = numpy_input_fn(x, y, shuffle=False)

    return input_fn
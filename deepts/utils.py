import numpy as np

import tensorflow as tf
from tensorflow.keras import Input, callbacks

from deepts.feature_column import SparseFeat, DenseFeat


def build_input_features(feature_columns):
    pass


def feature_columns_integer(feature_columns):
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
    dense_feature_dim = len(dense_feature_columns)
    sparse_feature_dim = 0
    for sfc in sparse_feature_columns:
        if sfc.embed_dim > 0:
            sparse_feature_dim += sfc.embed_dim
        else:
            sparse_feature_dim += sfc.vocab_size
    
    return sparse_feature_dim, dense_feature_dim

def input_from_feature_columns(feature_columns, window_size, lag):
    sparse_feature_dim, dense_feature_dim = feature_columns_integer(feature_columns)
    input_size = sparse_feature_dim + dense_feature_dim + lag
    inputs = Input(shape=(window_size, input_size,), name='Input_begin')
    return inputs

def numpy_input_fn(x, y, **kwargs):
    if tf.__version__ >= '2.0.0':
        return tf.compat.v1.estimator.inputs.numpy_input_fn(x, y, **kwargs)
    else:
        return tf.estimator.inputs.numpy_input_fn(x, y, **kwargs)

def variable_scope(name_or_scope):
    try:
        return tf.variable_scope(name_or_scope)
    except AttributeError:
        return tf.compat.v1.variable_scope(name_or_scope)

def root_mean_squared_error(labels, predictions, **kwargs):
    if tf.__version__ > '2.0.0':
        rmse, rmse_update = tf.compat.v1.metrics.root_mean_squared_error(labels, predictions, **kwargs)
    else:
        rmse, rmse_update = tf.metrics.root_mean_squared_error(labels, predictions, **kwargs)

def get_callbacks(config):
    if len(config) <= 0:
        return []
    callback_list = []
    if 'tensorboard' in config:
        config_tb = config['tensorboard']
        callback_list.append(callbacks.TensorBoard(**config_tb))
    elif 'earlystopping' in config:
        config_es = config['earlystopping']
        callback_list.append(callbacks.EarlyStopping(**config_es))

    return callback_list if len(callback_list) > 0 else None

def dataset_split(x, y, ratio_list=[6, 2, 2]):
    ratio_list = np.array([0.0] + ratio_list) / sum(ratio_list)
    index_list = list(map(int, np.cumsum(ratio_list) * len(y)))
    train_slice, valid_slice, test_slice = [slice(index_list[i], index_list[i+1]) \
                                            for i in range(len(index_list)-1)]
    return x[train_slice], y[train_slice],\
        x[valid_slice], y[valid_slice],\
        x[test_slice], y[test_slice]
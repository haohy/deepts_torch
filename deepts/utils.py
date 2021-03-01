
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import Input, callbacks

from deepts.feature_column import SparseFeat, DenseFeat

def set_logging():
    import logging
    logging.basicConfig(level = logging.INFO,
        format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
    return logging

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

def get_callbacks(config, tag):
    if len(config) <= 0:
        return []
    callback_list = []
    if 'tensorboard' in config:
        config_tb = config['tensorboard']
        log_dir = os.path.join(config_tb.pop('log_dir'), tag)
        callback_list.append(callbacks.TensorBoard(log_dir=log_dir, **config_tb))
    elif 'earlystopping' in config:
        config_es = config['earlystopping']
        callback_list.append(callbacks.EarlyStopping(**config_es))

    return callback_list if len(callback_list) > 0 else None


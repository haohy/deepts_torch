import os
import math
import numpy as np
import pandas as pd

import tensorflow as tf

from deepts.feature_column import SparseFeat, DenseFeat
from deepts.utils import get_callbacks

from IPython import embed

def get_raw_data(sample_size, num_sparse_feat, num_dense_feat):
    """Get raw input time series data. 
    
    :return: pd.DataFrame and pd.Series.
    """
    data_numpy_all = {}
    feature_columns = []
    for i in range(num_sparse_feat):
        vocab_size = np.random.randint(10, 20)
        embed_dim = 5
        data_numpy_all['sparse_' + str(i)] = np.random.randint(0, vocab_size, (sample_size,))
        feature_columns.append(SparseFeat('sparse_'+str(i), vocab_size, 'int64', embed_dim))
    for i in range(num_dense_feat):
        data_numpy_all['dense_' + str(i)] = np.random.randn(sample_size,)
        feature_columns.append(DenseFeat('dense_'+str(i), 1, 'float32'))
    data_numpy_all['target'] = np.random.randn(sample_size,)
    data_df = pd.DataFrame(data_numpy_all)
    Y = data_df.pop('target')
    X = data_df

    return X, Y, feature_columns

def get_input_fn(feature_dict, Y):
    pass

def check_model(model, model_name, x, y, batch_size, metrics):
    config_callbacks = {"tensorboard": {"log_dir": "/home/haohy/TSF/deepts/unittests/logs"}}
    callback_list = get_callbacks(config_callbacks)
    model.compile('adam', 'mae', metrics=metrics)
    model.fit(x, y, batch_size=batch_size, epochs=2, callbacks=callback_list)
    print("{} test train pass!".format(model_name))

    model.save_weights(model_name + '_weights.h5')
    model.load_weights(model_name + '_weights.h5')
    os.remove(model_name + '_weights.h5')
    print("{} test save and load pass!".format(model_name))

def check_estimator(estimator, input_fn):
    estimator.train(input_fn)
    estimator.evaluate(input_fn)
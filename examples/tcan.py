import json
import os, sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(1, os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import tensorflow as tf
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError

from deepts.models import TCAN
from deepts.metrics import MASE
from deepts.data import DeeptsData, dataset_split
from deepts.utils import get_callbacks
from examples.datasets import get_raw_df
from examples.utils import set_logging, save_predictions, plot_predictions

logging = set_logging()
from IPython import embed


def TSF_TCAN(config_model, config_dataset, model_name, ds_name):
    # configuration
    config = config_model[model_name]
    channel_list = config['channel_list']
    kernel_size = config['kernel_size']
    num_sub_blocks = config['num_sub_blocks']
    attn_dim = config['attn_dim']
    temp_attn = config['temp_attn']
    en_res = config['en_res']
    is_conv = config['is_conv']
    softmax_axis = config['softmax_axis']
    visual = config['visual']
    lag = config['lag']
    n_back = config['n_back']
    n_fore = config['n_fore']
    batch_size = config['batch_size']
    config_callbacks = config['callbacks']
    metrics = [MASE(), MeanAbsoluteError(), MeanAbsolutePercentageError()]

    # dataset, x: [batch_size, n_back, n_feature], y: [batch_size, 1, n_fore]
    X, Y, feature_columns = get_raw_df(config_dataset, ds_name)
    dataset = DeeptsData(X, Y, feature_columns, ds_name, n_back, n_fore, lag)
    # X, Y, feature_columns, name, n_back=1, n_fore=1, lag=0
    x, y = dataset.get_deepts_data()
    x_train, y_train, x_valid, y_valid, x_test, y_test = dataset_split(x, y)

    # model
    model = TCAN(feature_columns, n_back, n_fore, lag, attn_dim, channel_list, 
                num_sub_blocks, temp_attn, en_res, is_conv, softmax_axis, kernel_size, visual)
    print(model.summary())

    # train
    callback_list = get_callbacks(config_callbacks)
    model.compile(
        optimizer='adam', 
        loss='mae', 
        metrics=metrics
    )
    model.fit(
        x_train, 
        y_train, 
        batch_size=batch_size, 
        epochs=20, 
        callbacks=callback_list, 
        validation_data=(x_valid, y_valid)
    )
    y_pred = model.predict(x_test)

    # save results
    y_back_inverse = dataset.min_max_inverse_normalize(x_test[:, :, -lag].numpy())
    y_true_inverse = dataset.min_max_inverse_normalize(y_test.numpy())
    y_pred_inverse = dataset.min_max_inverse_normalize(y_pred)
    filename = save_predictions(y_back_inverse, y_true_inverse, y_pred_inverse)
    plot_predictions(filename, [0, int(len(y_pred)/2), -1])
    logging.info('Finished.')


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.set_visible_devices(physical_devices[1:], 'GPU')
    with open(os.path.join('/home/haohy/TSF/deepts', 'examples', "config.json"), 'r') as conf:
        config_all = json.load(conf)
    config_dataset = config_all['dataset']
    config_model = config_all['model']
    TSF_TCAN(config_model, config_dataset, 'TCAN', 'bike_hour')
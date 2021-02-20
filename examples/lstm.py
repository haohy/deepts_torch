import json
import os, sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(1, os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import tensorflow as tf

from deepts.models import LSTM
from deepts.estimator import LSTMEstimator
from deepts.metrics import MASE
from deepts.data import get_deepts_data, get_deepts_estimator_data
from deepts.utils import get_callbacks, dataset_split
from examples.datasets import get_raw_df
from examples.utils import set_logging, save_predictions, plot_predictions

logging = set_logging()
from IPython import embed


def TSF_LSTM(config_model, config_dataset, model_name, ds_name):
    # configuration
    config = config_model[model_name]
    hid_dim = config['hid_dim']
    lag = config['lag']
    n_back = config['n_back']
    n_fore = config['n_fore']
    batch_size = config['batch_size']
    config_callbacks = config['callbacks']

    # dataset, x: [batch_size, n_back, n_feature], y: [batch_size, 1, n_fore]
    X, Y, feature_columns = get_raw_df(config_dataset, ds_name)
    x, y = get_deepts_data(X, Y, feature_columns, n_back, n_fore, lag)
    x_train, y_train, x_valid, y_valid, x_test, y_test = dataset_split(x, y)

    # model
    model = LSTM(feature_columns, hid_dim, n_back, n_fore, lag)
    print(model.summary())

    # train
    callback_list = get_callbacks(config_callbacks)
    model.compile('adam', 'mae', metrics=[MASE()])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=100, callbacks=callback_list, 
            validation_data=(x_valid, y_valid))
    y_pred = model.predict(x_test)
    filename = save_predictions(x_test, y_test, y_pred)
    plot_predictions(filename, lag)
    logging.info('Finished.')


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.set_visible_devices(physical_devices[1:], 'GPU')
    with open(os.path.join('/home/haohy/TSF/deepts', 'examples', "config.json"), 'r') as conf:
        config_all = json.load(conf)
    config_dataset = config_all['dataset']
    config_model = config_all['model']
    TSF_LSTM(config_model, config_dataset, 'LSTM', 'bike_hour')
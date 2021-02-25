import json
import os, sys
import pickle

from tensorflow.keras import losses
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(1, os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError

from deepts.models import DeepTCN
from deepts.data import DeeptsData
from deepts.utils import get_callbacks
from deepts.metrics import MASE, ND, NRMSE
from examples.utils import set_logging, save_predictions, plot_predictions


logging = set_logging()
from IPython import embed


def TSF_DeepTCN(config_model, config_dataset, model_name, ds_name):
    config = config_model[model_name]
    lag = config['lag']
    n_back = config['n_back']
    n_fore = config['n_fore']
    batch_size = config['batch_size']
    config_callbacks = config['callbacks']
    metrics = [ND(), NRMSE()]

    with open(os.path.join(config_dataset['dir_root'], config_dataset[ds_name]['file_path']), 'rb') as f:
        trainX_dt, _, trainY_dt, trainY2_dt, testX_dt, _, testY_dt, testY2_dt = pickle.load(f)

    with open(os.path.join(config_dataset['dir_root'], config_dataset[ds_name]['scaler_file_path']), 'rb') as f:
        scaler = pickle.load(f)['standard_scaler']

    x = [trainX_dt, trainY2_dt]
    y = trainY_dt

    callback_list = get_callbacks(config_callbacks)
    model = DeepTCN(n_back, n_fore)
    print(model.summary())

    model.compile(
        optimizer='adam', 
        loss=[Huber()], 
        metrics=metrics,
    )
    model.fit(
        x, y, 
        batch_size=batch_size, 
        epochs=100,
        callbacks=callback_list
    )
    print("{} test train pass!".format(model_name))

    # dataset, x: [batch_size, n_back, n_feature], y: [batch_size, 1, n_fore]
    y_pred = model.predict([testX_dt, testY2_dt])

    # save results
    y_back_inverse = scaler.inverse_transform(testX_dt)
    y_true_inverse = testY_dt
    y_pred_inverse = y_pred
    filename = save_predictions(y_back_inverse, y_true_inverse, y_pred_inverse)
    plot_predictions(filename, [0, int(len(y_pred)/2), -1])
    logging.info('Finished.')


if __name__ == '__main__':
    with open(os.path.join('/home/haohy/TSF/deepts', 'examples', "config.json"), 'r') as conf:
        config_all = json.load(conf)
    config_dataset = config_all['dataset']
    config_model = config_all['model']
    TSF_DeepTCN(config_model, config_dataset, 'DeepTCN', 'electricity')
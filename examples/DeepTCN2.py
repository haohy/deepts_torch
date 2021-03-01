import json
import os, sys
import pickle
import datetime

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(1, os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError

from deepts.models import DeepTCN2
from deepts.data import TSDataset, get_dataloader
from deepts.utils import get_callbacks
from deepts.metrics import MASE, ND, NRMSE
from examples.utils import set_logging, save_predictions, plot_predictions

logging = set_logging()
from IPython import embed


def TSF_DeepTCN(config_model, config_dataset, model_name, ds_name):
    config = config_model[model_name]
    target = config_dataset[ds_name]['target']
    static_feat_col = config_dataset[ds_name]['static_feat']
    dynamic_feat_cat_dict = config_dataset[ds_name]['dynamic_feat_cat']
    dynamic_feat_real_col = config_dataset[ds_name]['dynamic_feat_real']
    pkl_path = os.path.join(config_dataset['dir_root'], config_dataset[ds_name]['pkl_path'])
    static_feat_dim = config_dataset[ds_name]['static_feat_dim']
    lag = config['lag']
    n_back = config['n_back']
    n_fore = config['n_fore']
    norm = config['norm']
    sliding_window_dis = config['sliding_window_dis']
    dilation_list = config['dilation_list']
    conv_ksize = config['conv_ksize']
    dilation_depth = config['dilation_depth']
    n_repeat = config['n_repeat']
    batch_size = config['batch_size']
    config_callbacks = config['callbacks']
    metrics = [ND(), NRMSE()]
    tag = model_name+'_'+ds_name+'_'+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    df = pd.read_csv(os.path.join(config_dataset['dir_root'], config_dataset[ds_name]['file_path']), 
                    **config_dataset[ds_name]['csv_kwargs'])

    dataset = TSDataset(df, target, static_feat_col, dynamic_feat_cat_dict, dynamic_feat_real_col, 
                n_back, n_fore, lag, sliding_window_dis, norm, pkl_path)

    x_train, y_train, x_valid, y_valid, x_test, y_test = get_dataloader(dataset, dynamic_feat_cat_dict,
        n_back, n_fore, static_feat_dim, dyn_feat_cat_kw='all')

    conv_filter = static_feat_dim + x_train[2].shape[-1] + 1
    model = DeepTCN2(n_back, n_fore, dilation_list, conv_filter, conv_ksize, dilation_depth, 
            n_repeat, x_train[1].shape[-1], x_train[2].shape[-1])

    callback_list = get_callbacks(config_callbacks, tag)
    print(model.summary())

    model.compile(
        optimizer='adam', 
        loss=[Huber()], 
        metrics=metrics,
    )

    model.fit(
        x_train, y_train, 
        batch_size=batch_size, 
        epochs=100,
        validation_data=(x_valid, y_valid),
        callbacks=callback_list
    )
    print("{} test train pass!".format(model_name))

    y_pred = model.predict(x_test)

    # save results
    y_back_inverse = dataset.scaler.inverse_transform(x_test[0].numpy())
    y_true_inverse = dataset.scaler.inverse_transform(y_test.numpy())
    y_pred_inverse = dataset.scaler.inverse_transform(y_pred)
    filename = save_predictions(y_back_inverse, y_true_inverse, y_pred_inverse, tag)
    plot_predictions(filename, [0, int(len(y_pred)/2), -1])
    logging.info('Finished.')


if __name__ == '__main__':
    with open(os.path.join('/home/haohy/TSF/deepts', 'examples', "config.json"), 'r') as conf:
        config_all = json.load(conf)
    config_dataset = config_all['dataset']
    config_model = config_all['model']
    TSF_DeepTCN(config_model, config_dataset, 'DeepTCN2', 'bike_hour')
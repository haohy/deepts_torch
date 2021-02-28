import json
import os, sys
import pickle

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(1, os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError

from deepts.models import DeepTCN
from deepts.layers import static_embedding, dynamic_feature_cat_embedding
from deepts.data import TSDataset, dataset_split
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
    conv_filter = static_feat_dim + 1
    conv_ksize = config['conv_ksize']
    dilation_depth = config['dilation_depth']
    n_repeat = config['n_repeat']
    batch_size = config['batch_size']
    config_callbacks = config['callbacks']
    metrics = [ND(), NRMSE()]

    df = pd.read_csv(os.path.join(config_dataset['dir_root'], config_dataset[ds_name]['file_path']), 
                    **config_dataset[ds_name]['csv_kwargs'])

    dataset = TSDataset(df, target, static_feat_col, dynamic_feat_cat_dict, dynamic_feat_real_col, 
                n_back, n_fore, lag, sliding_window_dis, norm, pkl_path)
    if not dataset.is_cached:
        dynamic_feature_cat = dataset.get_dynamic_feature_cat()
        static_feat_num_dict = dataset.get_static_feat_num_dict()
        time_series = dataset.get_time_series(period='all')
        dataset.save_pkl()
    else:
        dataset.load_pkl()
        dynamic_feature_cat = dataset.dynamic_feature_cat
        static_feat_num_dict = dataset.static_feat_num_dict
        time_series = dataset.time_series
    static_feat = static_embedding(static_feat_num_dict, n_back, n_fore, 'all', static_feat_dim)
    dynamic_feature_cat_embed = dynamic_feature_cat_embedding(dynamic_feature_cat, dynamic_feat_cat_dict)
    ts_back, ts_fore = time_series[:, :n_back], time_series[:, n_back:]
    ts_back_train, ts_back_valid, ts_back_test = dataset_split(ts_back)
    static_feat_train, static_feat_valid, static_feat_test = dataset_split(static_feat)
    dynamic_feature_cat_embed_train, dynamic_feature_cat_embed_valid, dynamic_feature_cat_embed_test\
         = dataset_split(dynamic_feature_cat_embed)
    ts_fore_train, ts_fore_valid, ts_fore_test = dataset_split(ts_fore)

    x_train = [ts_back_train, static_feat_train, dynamic_feature_cat_embed_train]
    y_train = ts_fore_train

    x_valid = [ts_back_valid, static_feat_valid, dynamic_feature_cat_embed_valid]
    y_valid = ts_fore_valid

    model = DeepTCN(n_back, n_fore, dilation_list, conv_filter, conv_ksize, dilation_depth, 
            n_repeat, static_feat.shape[-1], dynamic_feature_cat_embed.shape[-1])

    # with open(os.path.join(config_dataset['dir_root'], config_dataset[ds_name]['file_path']), 'rb') as f:
    #     trainX_dt, _, trainY_dt, trainY2_dt, testX_dt, _, testY_dt, testY2_dt = pickle.load(f)

    # with open(os.path.join(config_dataset['dir_root'], config_dataset[ds_name]['scaler_file_path']), 'rb') as f:
    #     scaler = pickle.load(f)['standard_scaler']

    # x = [trainX_dt, trainY2_dt]
    # y = trainY_dt

    callback_list = get_callbacks(config_callbacks)
    # model = DeepTCN(n_back, n_fore)
    print(model.summary())

    model.compile(
        optimizer='adam', 
        loss=[Huber()], 
        metrics=metrics,
    )
    model.fit(
        x_train, y_train, 
        batch_size=batch_size, 
        epochs=300,
        validation_data=(x_valid, y_valid),
        callbacks=callback_list
    )
    print("{} test train pass!".format(model_name))

    # dataset, x: [batch_size, n_back, n_feature], y: [batch_size, 1, n_fore]
    # y_pred = model.predict([testX_dt, testY2_dt])
    x_test = [ts_back_test, static_feat_test, dynamic_feature_cat_embed_test]
    y_test = ts_fore_test
    y_pred = model.predict(x_test)

    # save results
    y_back_inverse = dataset.scaler.inverse_transform(ts_back_test.numpy())
    y_true_inverse = dataset.scaler.inverse_transform(y_test.numpy())
    y_pred_inverse = y_pred
    filename = save_predictions(y_back_inverse, y_true_inverse, y_pred_inverse)
    plot_predictions(filename, [0, int(len(y_pred)/2), -1])
    logging.info('Finished.')


if __name__ == '__main__':
    with open(os.path.join('/home/haohy/TSF/deepts', 'examples', "config.json"), 'r') as conf:
        config_all = json.load(conf)
    config_dataset = config_all['dataset']
    config_model = config_all['model']
    TSF_DeepTCN(config_model, config_dataset, 'DeepTCN', 'demo')
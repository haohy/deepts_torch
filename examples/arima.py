import json
import os, sys
import datetime
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(1, os.getcwd())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from deepts.data import Data, TSDataset, dataset_split
from deepts.layers import static_embedding, dynamic_feature_cat_embedding
from deepts.metrics import MASE, ND, NRMSE, SMAPE, MAE, MSE
from examples.utils import set_logging, save_predictions, plot_predictions, record

logging = set_logging()
from IPython import embed


def TSF_SVR(config_model, config_dataset, model_name, ds_name):
    config = config_model[model_name]
    target = config_dataset[ds_name]['target']
    static_feat_col = config_dataset[ds_name]['static_feat']
    dynamic_feat_cat_dict = config_dataset[ds_name]['dynamic_feat_cat']
    dynamic_feat_real_col = config_dataset[ds_name]['dynamic_feat_real']
    pkl_path = os.path.join(config_dataset['dir_root'], config_dataset[ds_name]['pkl_path'])
    static_feat_dim = config_dataset[ds_name]['static_feat_dim']
    lag = config_dataset[ds_name]['lag']
    n_back = config_dataset[ds_name]['n_back']
    n_fore = config_dataset[ds_name]['n_fore']
    sliding_window_dis = config_dataset[ds_name]['sliding_window_dis']
    norm = config['norm']
    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    tag = model_name+'_'+ds_name+'_'+now

    df = pd.read_csv(os.path.join(config_dataset['dir_root'], config_dataset[ds_name]['file_path']), 
                    **config_dataset[ds_name]['csv_kwargs'])

    dataset = TSDataset(df, target, static_feat_col, dynamic_feat_cat_dict, dynamic_feat_real_col, 
                n_back, n_fore, lag, sliding_window_dis, norm, pkl_path)

    if not dataset.is_cached:
        dynamic_feature_cat = dataset.get_dynamic_feature_cat(period='all')
        dynamic_feature_real = dataset.get_dynamic_feature_real(period='all')
        static_feat_num_dict = dataset.get_static_feat_num_dict()
        time_series = dataset.get_time_series(period='all')
        dataset.save_pkl()
    else:
        dataset.load_pkl()
        dynamic_feature_cat = dataset.dynamic_feature_cat
        static_feat_num_dict = dataset.static_feat_num_dict
        dynamic_feature_real = dataset.dynamic_feature_real
        time_series = dataset.time_series

    dim_feat = dynamic_feature_real.shape[-1]
    x_train, x_test = train_test_split(dynamic_feature_real, test_size=0.2)
    y_train, y_test = train_test_split(time_series, test_size=0.2)
    x_train_re = np.reshape(x_train, [-1, dim_feat])
    x_test_re = np.reshape(x_test, [-1, dim_feat])
    y_train_re = np.reshape(y_train, [-1, 1])
    y_test_re = np.reshape(y_test, [-1, 1])


    embed(header="arima")


    regr = SVR()
    regr.fit(x_train_re, y_train_re)
    y_pred = regr.predict(x_test_re)
    y_pred_re = np.reshape(y_pred, [-1, 72])[:, -24:]
    y_test_re = np.reshape(y_test_re, [-1, 72])[:, -24:]

    results = {
        'model_name': 'SVR',
        'nd_test': round(float(ND(torch.tensor(y_test_re).clone().detach(), torch.tensor(y_pred_re))), 6),
        'smape_test': round(float(SMAPE(torch.tensor(y_test_re).clone().detach(), torch.tensor(y_pred_re))), 6),
        'nrmse_test': round(float(NRMSE(torch.tensor(y_test_re).clone().detach(), torch.tensor(y_pred_re))), 6),
        'mae_test': round(float(MAE(torch.tensor(y_test_re).clone().detach(), torch.tensor(y_pred_re))), 6),
        'mse_test': round(float(MSE(torch.tensor(y_test_re).clone().detach(), torch.tensor(y_pred_re))), 6)
    }
    record(config_dataset['record_file'], results)
    
    ts_index = [0, len(y_pred_re)//2, -1]
    x_fore = list(range(n_fore))
    fig, axes= plt.subplots(len(ts_index), 1, figsize=(8, 2+2*len(ts_index)))
    for idx, ts_idx in enumerate(ts_index):
        axes[idx].plot(x_fore, y_pred_re[ts_idx], label='pred')
        axes[idx].plot(x_fore, y_test_re[ts_idx], label='true')
        axes[idx].legend()
    plt.savefig('./examples/results/'+tag+'_svr.png', format='png')
    plt.close()

    logging.info('Finished.')


if __name__ == '__main__':
    with open(os.path.join('/home/haohy/TSF/deepts_torch', 'examples', "config.json"), 'r') as conf:
        config_all = json.load(conf)
    config_dataset = config_all['dataset']
    config_model = config_all['model']
    TSF_SVR(config_model, config_dataset, 'DeepTCN3', 'bike_hour_svr')
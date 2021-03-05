import json
import os, sys
import datetime
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(1, os.getcwd())

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data 
from torch.utils.tensorboard import SummaryWriter

from deepts.models import DeepTCN3
from deepts.dilate_loss import dilate_loss
from deepts.layers import static_embedding, dynamic_feature_cat_embedding
from deepts.data import Data, TSDataset, dataset_split
from deepts.metrics import MASE, ND, NRMSE
from examples.utils import set_logging, save_predictions, plot_predictions, record

logging = set_logging()
logger = SummaryWriter('./logs')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    epochs = config['epochs']
    sliding_window_dis = config['sliding_window_dis']
    dilation_list = config['dilation_list']
    conv_ksize = config['conv_ksize']
    dilation_depth = config['dilation_depth']
    n_repeat = config['n_repeat']
    batch_size = config['batch_size']
    config_callbacks = config['callbacks']
    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    tag = model_name+'_'+ds_name+'_'+now

    df = pd.read_csv(os.path.join(config_dataset['dir_root'], config_dataset[ds_name]['file_path']), 
                    **config_dataset[ds_name]['csv_kwargs'])

    dataset = TSDataset(df, target, static_feat_col, dynamic_feat_cat_dict, dynamic_feat_real_col, 
                n_back, n_fore, lag, sliding_window_dis, norm, pkl_path)

    if not dataset.is_cached:
        dynamic_feature_cat = dataset.get_dynamic_feature_cat(period='future')
        dynamic_feature_real = dataset.get_dynamic_feature_real(period='all')
        static_feat_num_dict = dataset.get_static_feat_num_dict()
        time_series = dataset.get_time_series(period='all')
        dataset.save_pkl()
    else:
        dataset.load_pkl()
        dynamic_feature_cat = dataset.dynamic_feature_cat
        static_feat_num_dict = dataset.static_feat_num_dict
        dynamic_feature_real = torch.FloatTensor(dataset.dynamic_feature_real)
        time_series = dataset.time_series
    static_feat = static_embedding(static_feat_num_dict, n_back, n_fore, 'all', static_feat_dim)
    dynamic_feature_cat_embed = dynamic_feature_cat_embedding(dynamic_feature_cat, dynamic_feat_cat_dict)
    ts_back, ts_fore = time_series[:, :n_back], time_series[:, n_back:]
    dynamic_feature_real_back = dynamic_feature_real[:, :n_back, :]

    ts_back_concat = torch.cat([torch.unsqueeze(ts_back, dim=-1), dynamic_feature_real_back], dim=-1)
    ts_back_train, ts_back_valid, ts_back_test = dataset_split(ts_back_concat)
    static_feat_train, static_feat_valid, static_feat_test = dataset_split(static_feat)
    dynamic_feature_cat_embed_train, dynamic_feature_cat_embed_valid, dynamic_feature_cat_embed_test\
        = dataset_split(dynamic_feature_cat_embed)
    ts_fore_train, ts_fore_valid, ts_fore_test = dataset_split(ts_fore)

    x_train = [ts_back_train, static_feat_train, dynamic_feature_cat_embed_train]
    y_train = ts_fore_train

    x_valid = [ts_back_valid, static_feat_valid, dynamic_feature_cat_embed_valid]
    y_valid = ts_fore_valid
    
    x_test = [ts_back_test, static_feat_test, dynamic_feature_cat_embed_test]
    y_test = ts_fore_test

    train_dataloader = data.DataLoader(Data(x_train, y_train), batch_size)
    valid_dataloader = data.DataLoader(Data(x_valid, y_valid), batch_size)
    test_dataloader = data.DataLoader(Data(x_test, y_test), batch_size)

    conv_filter = static_feat_dim + ts_back_concat.shape[-1]
    model = DeepTCN3(n_back, n_fore, dilation_list, conv_filter, conv_ksize, dilation_depth, 
            n_repeat, ts_back_concat.shape[-1], x_train[1].shape[-1], x_train[2].shape[-1]+x_train[1].shape[-1])
    num_parameters_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # criterion = nn.MSELoss()
    criterion = dilate_loss
    criterion_mse = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs+1):
        loss = 0
        for [input_ts, input_static_emb, input_dynamic_emb], label in train_dataloader:
            optimizer.zero_grad()
            input_ts = input_ts
            input_static_emb = input_static_emb
            input_dynamic_emb = input_dynamic_emb
            y_pred = model(input_ts, input_static_emb, input_dynamic_emb)
            loss_i = criterion(y_pred, torch.unsqueeze(label, dim=-1), device=device)[0]
            
            loss += loss_i.item()
            loss_i.backward()
            optimizer.step()
        
        y_pred_valid = model(*x_valid)
        loss_val = criterion(y_pred_valid, torch.unsqueeze(y_valid, dim=-1), device=device)[0]
        nd_valid = ND(y_valid.cpu().detach(), torch.squeeze(y_pred_valid, dim=-1).cpu().detach())
        nrmse_valid = NRMSE(y_valid.cpu().detach(), torch.squeeze(y_pred_valid, dim=-1).cpu().detach())
        logger_note = tag
        logger.add_scalars('{}/loss_train'.format(logger_note), {'loss_train': loss}, epoch)
        logger.add_scalars('{}/loss_val'.format(logger_note), {'loss_val':loss_val}, epoch) 
        logger.add_scalars('{}/nd_valid'.format(logger_note), {'nd_valid': nd_valid}, epoch)
        logger.add_scalars('{}/nrmse_valid'.format(logger_note), {'nrmse_valid':nrmse_valid}, epoch) 
        print("Epoch {}, Train Loss {:.4f}, Valid ND {:.4f}, Valid NRMSE {:.4f}, Valid loss {:.4f}"\
            .format(epoch, loss/len(train_dataloader), nd_valid, nrmse_valid, loss_val.item()))

    # dataset, x: [batch_size, n_back, n_feature], y: [batch_size, 1, n_fore]
    # y_pred = model.predict([testX_dt, testY2_dt])
    
    model.eval()
    y_pred = model(*x_test)

    # save results
    y_back_inverse = dataset.scaler.inverse_transform(x_test[0][:,:,0].numpy())
    y_true_inverse = dataset.scaler.inverse_transform(y_test.numpy())
    y_pred_inverse = dataset.scaler.inverse_transform(torch.squeeze(y_pred, dim=-1).cpu().detach().numpy())
    filename = save_predictions(y_back_inverse, y_true_inverse, y_pred_inverse, tag)
    plot_predictions(filename, [0, int(len(y_pred)/2), -1])

    note = "test reconstruct."
    config.update({'datetime': now,
        'num_params': num_parameters_train,
        'nd_valid': round(loss_val.item(), 6),
        'nrmse_valid': round(float(nrmse_valid), 6),
        'nd_test': round(float(ND(y_test.cpu().detach(), torch.squeeze(y_pred, dim=-1))), 6),
        'nrmse_test': round(float(NRMSE(y_test.cpu().detach(), torch.squeeze(y_pred, dim=-1))), 6),
        'note': note})
    record(config_dataset['record_file'], config)
    logging.info('Finished.')


if __name__ == '__main__':
    with open(os.path.join('/home/haohy/TSF/deepts_torch', 'examples', "config.json"), 'r') as conf:
        config_all = json.load(conf)
    config_dataset = config_all['dataset']
    config_model = config_all['model']
    TSF_DeepTCN(config_model, config_dataset, 'DeepTCN3', 'bike_hour_deeptcn3')
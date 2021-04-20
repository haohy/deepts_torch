import json
import time
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
import torch.multiprocessing as mp

from deepts.models import DeepTCN3
from deepts.dilate_loss import dilate_loss
from deepts.multi_scale_dtw_loss import multi_scale_dtw_loss
from deepts.layers import static_embedding, dynamic_feature_cat_embedding
from deepts.data import Data, TSDataset, dataset_split
from deepts.metrics import MASE, ND, NRMSE, SMAPE, MAE, MSE
from examples.utils import set_logging, save_predictions, plot_predictions, record, draw_attn, save_model, load_model

logging = set_logging()
logger = SummaryWriter('./examples/logs')
from IPython import embed

def evaluate(model, x_test_list, y_test, multi_scale_dtw_loss, scale_all, gamma, device, final=False):
    model.eval()
    input_ts, input_static_emb, input_dynamic_emb = x_test_list
    input_ts = input_ts.to(device)
    input_static_emb = input_static_emb.to(device)
    input_dynamic_emb = input_dynamic_emb.to(device)
    y_test = y_test.to(device)
    ms_dtw_test = 0.0
    with torch.no_grad():
        y_pred_test, _ = model(input_ts, input_static_emb, input_dynamic_emb)
        nd_test = float(ND(y_test, y_pred_test))
        smape_test = float(SMAPE(y_test, y_pred_test))
        nrmse_test = float(NRMSE(y_test, y_pred_test))
        mae_test = float(MAE(y_test, y_pred_test))
        mse_test = float(MSE(y_test, y_pred_test))
        if final:
            ms_dtw_test = float(multi_scale_dtw_loss(y_pred_test, y_test, scale_all, 1.0, 0.0001).item())

    return nd_test, smape_test, nrmse_test, mae_test, mse_test, ms_dtw_test, y_pred_test

def TSF_DeepTCN(config_model, config_dataset, model_name, ds_name, criterion_type, 
                alpha=0.5, gamma=0.01, num_scales=5, gpu_id=3):
    # device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
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
    epochs = config['epochs']
    dilation_list = config['dilation_list']
    out_features = config['out_features']
    conv_ksize = config['conv_ksize']
    hid_dim_fore = config['hid_dim_fore']
    nheads = config['nheads']
    dilation_depth = config['dilation_depth']
    n_repeat = config['n_repeat']
    batch_size = config['batch_size']
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
        # dynamic_feature_real = torch.FloatTensor(dataset.dynamic_feature_real)
        time_series = dataset.time_series
    static_feat = static_embedding(static_feat_num_dict, n_back, n_fore, 'all', static_feat_dim)
    dynamic_feature_cat_embed = dynamic_feature_cat_embedding(dynamic_feature_cat, dynamic_feat_cat_dict)
    ts_back, ts_fore = time_series[:, :n_back], time_series[:, n_back:]
    # dynamic_feature_real_back = dynamic_feature_real[:, :n_back, :]
    ts_back_concat = torch.unsqueeze(ts_back, -1)
    # ts_back_concat = torch.cat([torch.unsqueeze(ts_back, dim=-1), dynamic_feature_real_back], dim=-1)
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

    ts_dim = x_train[0].shape[-1]
    static_dim = x_train[1].shape[-1]
    dynamic_dim = x_train[2].shape[-1]
    print("ts_dim: {}, static_dim: {}, dynamic_dim: {}".format(ts_dim, static_dim, dynamic_dim))
    model = DeepTCN3(n_back, n_fore, dilation_list, out_features, hid_dim_fore, conv_ksize, nheads, 
                    dilation_depth, n_repeat, ts_dim, static_dim, dynamic_dim)
    num_parameters_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("num_params: {}".format(num_parameters_train))

    patience_len = 50
    lr = 0.005
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = dilate_loss
    # num_scales = 3
    scale_list = list(range(n_fore))[-num_scales:]
    scale_all = list(range(n_fore))
    # scale_list = list(range(n_fore))[slice(1, n_fore, n_fore//num_scales)]
    
    criterion_dict = {'ms_dtwi': multi_scale_dtw_loss,
                    'mseloss': nn.MSELoss(),
                    'huberloss': nn.SmoothL1Loss()}
    criterion = criterion_dict[criterion_type]
    # if criterion_type == 'ms_dtwi':
    #     epochs = 300
    # else:
    #     epochs = 70
    

    model.train()
    model.to(device)
    loss_val_best = 1e8
    loss_val_list = []
    ave_time_list = []
    nd_test, smape_test, nrmse_test, mae_test, mse_test = 1e8, 1e8, 1e8, 1e8, 1e8

    # x_valid_list = [x.to(device) for x in x_valid]
    # y_valid = y_valid.to(device)
    # x_test_list = [x.to(device) for x in x_test]
    # y_test = y_test.to(device)

    for epoch in range(1, epochs+1):
        loss = 0
        last_pred = None
        attn_matrixes = None
        time_cost = []

        for [input_ts, input_static_emb, input_dynamic_emb], label in train_dataloader:
            # start_time = time.time()
            optimizer.zero_grad()
            input_ts = input_ts.to(device)
            input_static_emb = input_static_emb.to(device)
            input_dynamic_emb = input_dynamic_emb.to(device)
            label = label.to(device)
            y_pred, attn_matrixes = model(input_ts, input_static_emb, input_dynamic_emb)
            if criterion_type == 'ms_dtwi':
                loss_i = criterion(y_pred, label, scale_list, alpha, gamma)
            else:
                loss_i = criterion(y_pred, label)
            
            loss += loss_i.item()
            loss_i.backward()
            optimizer.step()

            # time_cost.append(time.time() - start_time)

            last_pred = y_pred
        # if attn_matrixes:
        #     draw_attn(attn_matrixes, epoch, tag)
        # print(last_pred[0, :10])
        # embed(header="dilate loss")
        # average_time = sum(time_cost) / len(train_dataloader)
        # ave_time_list.append(average_time)
        
        # y_pred_valid, _ = model(*x_valid_list)
        nd_valid, smape_valid, nrmse_valid, mae_valid, mse_valid, ms_dtw_valid, y_pred_valid = evaluate(model, x_valid, y_valid, multi_scale_dtw_loss, scale_all, gamma, device)
        y_valid = y_valid.to(device)
        if criterion_type == 'ms_dtwi':
            loss_val =  criterion(y_pred_valid, y_valid, scale_list, alpha, gamma)
        else:
            loss_val = criterion(y_pred_valid, y_valid)
        # nd_valid = ND(y_valid.cpu().detach(), y_pred_valid.cpu().detach())
        # smape_valid = SMAPE(y_valid.cpu().detach(), y_pred_valid.cpu().detach())
        # nrmse_valid = NRMSE(y_valid.cpu().detach(), y_pred_valid.cpu().detach())
        logger_note = tag
        logger.add_scalars('{}/loss_train'.format(logger_note), {'loss_train': loss/len(train_dataloader)}, epoch)
        logger.add_scalars('{}/loss_val'.format(logger_note), {'loss_val':loss_val.item()}, epoch) 
        logger.add_scalars('{}/nd_valid'.format(logger_note), {'nd_valid': nd_valid}, epoch)
        logger.add_scalars('{}/smape_valid'.format(logger_note), {'smape_valid': smape_valid}, epoch)
        logger.add_scalars('{}/nrmse_valid'.format(logger_note), {'nrmse_valid':nrmse_valid}, epoch) 
        print("Epoch {}, Train Loss {:.4f}, Valid ND {:.4f}, Valid SMAPE {:.4f}, Valid NRMSE {:.4f}, Valid loss {:.4f}"\
            .format(epoch, loss/len(train_dataloader), nd_valid, smape_valid, nrmse_valid, loss_val.item()))

        if epoch > patience_len and loss_val >= max(loss_val_list[-patience_len:]):
            lr = lr / 2.
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        loss_val_list.append(loss_val)
        nd_test, smape_test, nrmse_test, mae_test, mse_test, _, y_pred_test = evaluate(model, x_test, y_test, criterion, scale_all, gamma, device)

        if loss_val < loss_val_best:
            loss_val_best = loss_val
            y_pred_test_best = y_pred_test
            # save_model(model, tag)
    nd_test, smape_test, nrmse_test, mae_test, mse_test, ms_dtw_test, y_pred_test = evaluate(model, x_test, y_test, multi_scale_dtw_loss, scale_all, gamma, device, True)

    # save results
    y_back_inverse = dataset.scaler.inverse_transform(x_test[0][:,:,0].cpu().numpy())
    y_true_inverse = dataset.scaler.inverse_transform(y_test.cpu().numpy())
    y_pred_inverse = dataset.scaler.inverse_transform(y_pred_test_best.cpu().detach().numpy())

    note = "TCAN, " + ds_name +' '+ criterion_type +' '+ str(num_scales) +' '+ str(alpha) +' '+ str(gamma)
    # note = "TCAN, ms_dtwi"
    config.update({'ds_name': ds_name})
    config.update(config_dataset[ds_name])
    config.update({'datetime': now,
        'num_params': num_parameters_train,
        'nd_valid': round(float(nd_valid), 6),
        'smape_valid': round(float(smape_valid), 6),
        'nrmse_valid': round(float(nrmse_valid), 6),
        'mae_valid': round(mae_valid, 6),
        'mse_valid': round(mse_valid, 6),
        'ms_dtw_valid': round(ms_dtw_valid, 6),
        'nd_test': round(nd_test, 6),
        'smape_test': round(smape_test, 6),
        'nrmse_test': round(nrmse_test, 6),
        'mae_test': round(mae_test, 6),
        'mse_test': round(mse_test, 6),
        # 'nd_test_bm': round(nd_test_bm, 6),
        # 'smape_test_bm': round(smape_test_bm, 6),
        # 'nrmse_test_bm': round(nrmse_test_bm, 6),
        # 'mae_test_bm': round(mae_test_bm, 6),
        # 'mse_test_bm': round(mse_test_bm, 6),
        'ms_dtw_test': round(ms_dtw_test, 6),
        # 'time': sum(ave_time_list)/epochs,
        'note': note})
    record(config_dataset['record_file'], config)

    filename = save_predictions(y_back_inverse, y_true_inverse, y_pred_inverse, tag)
    plot_predictions(filename, [0, int(len(y_pred_test)/2), -1])
    logging.info('Finished.')


if __name__ == '__main__':
    with open(os.path.join('/home/haohy/TSF/deepts_torch', 'examples', "config.json"), 'r') as conf:
        config_all = json.load(conf)
    config_dataset = config_all['dataset']
    config_model = config_all['model']
    gamma_list = [0.0001, 0.001, 0.01, 0.1]
    alpha_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    num_scale_list = [2, 4, 6, 8, 10]
    # ds_name_list = ['traffic', 'TAS2016', 'bike_hour', 'PRSA']
    ds_name_list = ['bike_hour']
    criterion_type_list = ['huberloss', 'mseloss', 'ms_dtwi']
    # TSF_DeepTCN(config_model, config_dataset, 'DeepTCN3', 'bike_hour', 'ms_dtwi')
    # args_list = []
    # for ds_name in ds_name_list:
    #     for criterion_type in criterion_type_list:
    #         for alpha in alpha_list:
    #             for num_scale in num_scale_list:
                    # args_list.append([config_model, config_dataset, 'DeepTCN3', ds_name, criterion_type, alpha, num_scale])
    # for gamma in gamma_list:
    #     TSF_DeepTCN(config_model, config_dataset, 'DeepTCN3', 'bike_hour', 'ms_dtwi', 0.0, gamma, 4)
    # TSF_DeepTCN(config_model, config_dataset, 'DeepTCN3', 'bike_hour', 'ms_dtwi', 0.4, 0.01, 10)
    TSF_DeepTCN(config_model, config_dataset, 'DeepTCN3', 'bike_hour', 'mseloss', 0.4, 0.01, 10)
    # processes = []
   

    # num_processes = len(args_list)

    # for i in range(num_processes):
    #     p = mp.Process(target=TSF_DeepTCN, args=([*args_list[i]]))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()

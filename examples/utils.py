import os
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

save_dir = './examples/results/'

from IPython import embed

def set_logging():
    import logging
    logging.basicConfig(level = logging.INFO,
        format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
    return logging

logging = set_logging()

def save_predictions(y_back, y_true, y_pred, tag):
    test = {
        'y_back': y_back,
        'y_true': y_true,
        'y_pred': y_pred
    }
    
    filename = save_dir + tag + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(test, f)
    logging.info('Save predicitons!')
    return filename

def load_predictions(filename):
    with open(filename, 'rb') as f:
        testset = pickle.load(f)
    logging.info('Load predicitons!')
    return testset

def plot_predictions(filename, ts_index):
    fig, axes= plt.subplots(len(ts_index), 1, figsize=(8, 2+2*len(ts_index)))
    testset = load_predictions(filename)
    n_back = len(testset['y_back'][0])
    n_fore = len(testset['y_pred'][0])
    x_back = list(range(n_back))
    x_fore = list(range(n_back, n_back + n_fore))
    for idx, ts_idx in enumerate(ts_index):
        axes[idx].plot(x_back, testset['y_back'][ts_idx], label='back')
        axes[idx].plot(x_fore, testset['y_pred'][ts_idx], label='pred')
        axes[idx].plot(x_fore, testset['y_true'][ts_idx], label='true')
        axes[idx].legend()
    plt.savefig(filename+'.png', format='png')
    plt.close()
    logging.info("Plot predictions and the figure saved!")
    
def record(filename, record_dict):
    if os.path.isfile(filename):
        df = pd.read_csv(filename, header=0)
    else:
        df = pd.DataFrame()
    df = df.append(record_dict, ignore_index=True)
    df.to_csv(filename, index=False)
    logging.info("Saved results.")

def draw_attn(visual_info, epoch, tag):
    # visual_info: [3, n_back, n_back]
    dir_root = "/home/haohy/TSF/deepts_torch/examples/results/pngs/" + tag
    n_back = len(visual_info[0])
    if not os.path.isdir(dir_root):
        os.system('mkdir {}'.format(dir_root))
        for ts_idx in range(len(visual_info)):
            os.system('mkdir {}'.format(dir_root+'/'+str(ts_idx)))

    for ts_idx in range(3):
        plt.figure()
        plt.imshow(visual_info[ts_idx])
        new_ticks = np.append([0],np.arange(4, n_back, 5))
        plt.xticks(new_ticks, new_ticks+1)
        plt.yticks(new_ticks, new_ticks+1)
        plt.colorbar()
        plt.savefig(dir_root+'/'+str(ts_idx)+'/'+str(epoch)+'.png', format='png')
        plt.close()
    logging.info("Draw attention picture.")

def save_model(model, tag):
    # with open(args.dir_model+'/'+args.log+"_model.pt", 'wb') as f:
    #     torch.save(model, f)
    model_name = '/home/haohy/TSF/deepts_torch/examples/models/'+tag+"_model.pt"
    torch.save({'state_dict': model.state_dict()}, model_name)
    logging.info('Save model!')

def load_model(model, tag):
    # with open(args.dir_model+'/'+args.log+"_model.pt", 'rb') as f:
    #     model = torch.load(f)
    model_name = '/home/haohy/TSF/deepts_torch/examples/models/'+tag+"_model.pt"
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['state_dict'])
    return model
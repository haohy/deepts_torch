import os
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

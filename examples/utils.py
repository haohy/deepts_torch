import os
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt

save_dir = './examples/results/'

from IPython import embed

def set_logging():
    import logging
    logging.basicConfig(level = logging.INFO,
        format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
    return logging

def save_predictions(x_test, y_true, y_pred):
    logging = set_logging()
    test = {
        'x_test': x_test,
        'y_true': y_true,
        'y_pred': y_pred
    }
    tag = str(datetime.datetime.now())
    filename = save_dir + tag + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(test, f)
    logging.info('Save predicitons!')
    return filename

def load_predictions(filename):
    logging = set_logging()
    with open(filename, 'rb') as f:
        testset = pickle.load(f)
    logging.info('Load predicitons!')
    return testset

def plot_predictions(filename, lag):
    logging = set_logging()
    fig, axs= plt.subplots(1, 1)
    testset = load_predictions(filename)
    x_back = list(range(len(testset['x_test'][0])))
    x_fore = list(range(len(testset['x_test'][0]), len(testset['x_test'][0])+len(testset['y_pred'][0])))
    axs.plot(x_back, testset['x_test'][0][:, -lag], label='back')
    axs.plot(x_fore, testset['y_pred'][0], label='pred')
    axs.plot(x_fore, testset['y_true'][0], label='true')
    axs.legend()
    plt.savefig(filename+'.png', format='png')
    plt.close()
    logging.info("Plot predictions and the figure saved!")
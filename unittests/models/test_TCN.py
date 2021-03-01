import pytest
import os, sys
import pickle

from tensorflow.keras import losses
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(1, os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.losses import Huber

from deepts.models import TCN
from deepts.metrics import MASE, ND, NRMSE
from unittests.config import (SAMPLE_SIZE, MODEL_DIR, LAG, BATCH_SIZE, 
                            WINDOW_SIZE, N_BACK, N_FORE)

from IPython import embed


def test_TCN():
    model_name = 'TCN'
    metrics = [ND(), NRMSE()]

    with open('/home/haohy/TSF/deepts/examples/data/electricity/NewTCNQuantile/feature_prepare.pkl', 'rb') as f:
        trainX_dt, _, trainY_dt, trainY2_dt, testX_dt, _, testY_dt, testY2_dt = pickle.load(f)
    
    with open('/home/haohy/TSF/deepts/examples/data/electricity/NewTCNQuantile/electricity_standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)['standard_scaler']

    x = [trainX_dt, trainY2_dt]
    y = trainY_dt

    model = TCN()
    print(model.summary())

    model.compile(
        optimizer='adam', 
        loss=[Huber()], 
        metrics=metrics
    )
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=5)
    print("{} test train pass!".format(model_name))
    
    x_test = [testX_dt, testY2_dt]
    y_test = testX_dt

    embed(header="predict")
    y_pred = model.predict(x_test)

if __name__ == '__main__':
    # physical_devices = tf.config.list_physical_devices('GPU') 
    # tf.config.set_visible_devices(physical_devices[1:], 'GPU')
    test_TCN()
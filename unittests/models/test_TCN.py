import pytest
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
from deepts.metrics import MASE, ND, NRMSE
from unittests.utils import check_model, get_raw_data
from unittests.config import (SAMPLE_SIZE, MODEL_DIR, LAG, BATCH_SIZE, 
                            WINDOW_SIZE, N_BACK, N_FORE)

from IPython import embed

# @pytest.mark.parametrize(
#     'temp_attn, en_res, is_conv, softmax_axis',
#     [(True, True, True, 1),
#     (False, False, True, 1),
#     (True, True, False, 1),]
# )
def test_TCN():
    model_name = 'TCN'
    metrics = [ND(), NRMSE()]

    with open('/home/haohy/TSF/deepts/examples/data/electricity/NewTCNQuantile/feature_prepare.pkl', 'rb') as f:
        trainX_dt, _, trainY_dt, trainY2_dt, testX_dt, _, testY_dt, testY2_dt = pickle.load(f)

    x = [trainX_dt, trainY2_dt]
    y = trainY_dt

    model = DeepTCN()
    print(model.summary())

    model.compile(
        optimizer='adam', 
        loss=[Huber()], 
        metrics=metrics
    )
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=100)
    print("{} test train pass!".format(model_name))


if __name__ == '__main__':
    # physical_devices = tf.config.list_physical_devices('GPU') 
    # tf.config.set_visible_devices(physical_devices[1:], 'GPU')
    test_TCN()
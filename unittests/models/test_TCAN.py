import pytest
import os, sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(1, os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError

from deepts.models import TCAN
from deepts.data import DeeptsData
from deepts.metrics import MASE
from unittests.utils import check_model, get_raw_data
from unittests.config import (SAMPLE_SIZE, MODEL_DIR, LAG, BATCH_SIZE, 
                            WINDOW_SIZE, N_BACK, N_FORE)


@pytest.mark.parametrize(
    'temp_attn, en_res, is_conv, softmax_axis',
    [(True, True, True, 1),
    (False, False, True, 1),
    (True, True, False, 1),]
)
def test_TCAN(temp_attn, en_res, is_conv, softmax_axis):
    model_name = 'TCAN'
    channel_list = [20, 30, 40]
    kernel_size = 3
    num_sub_blocks = 2
    attn_dim = 10
    visual = False
    num_sparse_feat = 5
    num_dense_feat = 3
    metrics = [MASE(), MeanAbsoluteError(), MeanAbsolutePercentageError()]

    X, Y, feature_columns = get_raw_data(sample_size=SAMPLE_SIZE, num_sparse_feat=num_sparse_feat, 
                                        num_dense_feat=num_dense_feat)
    dataset = DeeptsData(X, Y, feature_columns, 'test', N_BACK, N_FORE, LAG)
    x, y = dataset.get_deepts_data()
    
    model = TCAN(feature_columns, N_BACK, N_FORE, LAG, attn_dim, channel_list, 
                num_sub_blocks, temp_attn, en_res, is_conv, softmax_axis, kernel_size, visual)
    print(model.summary())

    check_model(model, model_name, x, y, BATCH_SIZE, metrics)


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.set_visible_devices(physical_devices[1:], 'GPU')
    # test_TCAN(True, True, True, 1)
    pytest.main(['-s', './unittests/models/test_TCAN.py'])
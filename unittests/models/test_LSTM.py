import pytest
import os, sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(1, os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from deepts.models import LSTM
from deepts.estimator import LSTMEstimator
from deepts.data import DeeptsData
from deepts.metrics import MASE
from unittests.utils import check_model, check_estimator, get_raw_data
from unittests.config import SAMPLE_SIZE, MODEL_DIR, LAG, BATCH_SIZE, WINDOW_SIZE, N_BACK, N_FORE


@pytest.mark.parametrize(
    'num_sparse_feat, num_dense_feat',
    [(5, 3), (3, 2)]
)
def test_LSTM(num_sparse_feat, num_dense_feat):
    model_name = 'LSTM'
    hid_dim = 64
    metrics = [MASE()]

    X, Y, feature_columns = get_raw_data(sample_size=SAMPLE_SIZE, num_sparse_feat=num_sparse_feat, 
                                        num_dense_feat=num_dense_feat)
    dataset = DeeptsData(X, Y, feature_columns, 'test', N_BACK, N_FORE, LAG)
    x, y = dataset.get_deepts_data()

    model = LSTM(feature_columns, hid_dim, N_BACK, N_FORE, LAG)
    print(model.summary())

    check_model(model, model_name, x, y, BATCH_SIZE, metrics)


# def LSTMEstimator_test():
#     hid_dim = 64

#     estimator = LSTMEstimator(MODEL_DIR, hid_dim, lr=1e-3, config=None)

#     X, Y, feature_columns = get_raw_data(sample_size=SAMPLE_SIZE, num_sparse_feat=5, num_dense_feat=3)
#     input_fn = get_deepts_estimator_data(X, Y, feature_columns, window_size=WINDOW_SIZE, n_back=N_BACK, 
#                                         n_fore=N_FORE, lag=LAG, batch_size=BATCH_SIZE)

#     check_estimator(estimator, input_fn)


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.set_visible_devices(physical_devices[1:], 'GPU')
    pytest.main(['-s', './unittests/models/test_LSTM.py'])
    # test_LSTM(5, 3)
    # deeptsEstimator_test()

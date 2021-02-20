import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers

from deepts.layers import LSTMLayer
from deepts.utils import input_from_feature_columns

def LSTM(feature_columns, hid_dim, n_back, n_fore, lag):
    # h = tf.random.normal((1, hid_dim))
    # c = tf.random.normal((1, hid_dim))
    inputs = input_from_feature_columns(feature_columns, n_back, lag)
    # lstm_layer = LSTMLayer(hid_dim)
    lstm_layer = layers.LSTM(hid_dim)
    fc_layer = layers.Dense(n_fore)
    outputs = fc_layer(lstm_layer(inputs))
    
    return Model(inputs=inputs, outputs=outputs)
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers

from deepts.layers import LSTMLayer, LSTM2Layer
from deepts.utils import input_from_feature_columns

def LSTM(feature_columns, hid_dim, n_back, n_fore, lag):
    inputs = input_from_feature_columns(feature_columns, n_back, lag)
    lstm_layer = layers.LSTM(hid_dim)
    # lstm_layer = LSTMLayer(hid_dim)
    # lstm_layer = LSTM2Layer(hid_dim)
    fc_layer = layers.Dense(n_fore)
    # hs = tf.cast(tf.stack(lstm_layer(inputs)), dtype=tf.float32)
    hs = lstm_layer(inputs)
    outputs = fc_layer(hs)
    
    return Model(inputs=inputs, outputs=outputs)
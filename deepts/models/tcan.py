import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers

from deepts.layers import TemporalConvNet
from deepts.utils import input_from_feature_columns

def TCAN(feature_columns, n_back, n_fore, lag, attn_dim, channel_list, 
        num_sub_blocks, temp_attn, en_res, is_conv, softmax_axis, kernel_size, visual):
    inputs = input_from_feature_columns(feature_columns, n_back, lag)
    tcan_layer = TemporalConvNet(attn_dim, channel_list, num_sub_blocks, temp_attn, 
                                en_res, is_conv, softmax_axis, kernel_size, visual)
    fc_layer = layers.Dense(n_fore)
    hs, _ = tcan_layer(inputs)
    outputs = fc_layer(hs[:, -1, :])
    tf.print('+'*15, outputs.shape)

    return Model(inputs=inputs, outputs=outputs)
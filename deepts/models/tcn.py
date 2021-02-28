import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers, Input

from deepts.layers import TCN

def DeepTCN(n_back, n_fore, dilation_list, conv_filter, conv_ksize, dilation_depth, 
            n_repeat, static_dim, dynamic_dim):
    input_ts = Input(shape=[n_back,])
    input_static_emb = Input(shape=[n_back+n_fore, static_dim])
    input_dynamic_emb = Input(shape=[n_fore, dynamic_dim])
    tcn_layer = TCN(n_back, n_fore, dilation_list, conv_filter, conv_ksize, 
                    dilation_depth, n_repeat)
    outputs = tcn_layer(input_ts, input_static_emb, input_dynamic_emb)

    return Model(inputs=[input_ts, input_static_emb, input_dynamic_emb], outputs=outputs)
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers, Input

from deepts.layers import TCN
from deepts.utils import input_from_feature_columns

def DeepTCN():
    input1 = Input(shape=[168,])
    input2 = Input(shape=[24, 22])
    tcn_layer = TCN()
    outputs = tcn_layer(input1, input2)

    return Model(inputs=[input1, input2], outputs=outputs)
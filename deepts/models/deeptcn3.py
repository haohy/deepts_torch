import numpy as np
import pandas as pd
from torch import nn

from deepts.layers import DeepTCN3Layer


class DeepTCN3(nn.Module):
    def __init__(self, n_back, n_fore, dilation_list, out_features, hid_dim_fore, conv_ksize, nheads,
                dilation_depth, n_repeat, ts_dim, static_dim, dynamic_dim):
        super(DeepTCN3, self).__init__()
        self.tcn_layer = DeepTCN3Layer(ts_dim, static_dim, dynamic_dim, n_back, n_fore, dilation_list, out_features, 
                                    hid_dim_fore, conv_ksize, nheads, dilation_depth, n_repeat)
    
    def forward(self, input_ts, input_static_emb, input_dynamic_emb):
        return self.tcn_layer(input_ts, input_static_emb, input_dynamic_emb)
import numpy as np
import pandas as pd
from torch import nn

from deepts.layers import DeepTCN3Layer

# def DeepTCN3(n_back, n_fore, dilation_list, conv_filter, conv_ksize, dilation_depth, 
#             n_repeat, ts_dim, static_dim, dynamic_dim):
# #     input_ts = Input(shape=[n_back, ts_dim])
# #     input_static_emb = Input(shape=[n_back+n_fore, static_dim])
# #     input_dynamic_emb = Input(shape=[n_fore, dynamic_dim])
#     tcn_layer = DeepTCN3Layer(n_back, n_fore, dilation_list, conv_filter, conv_ksize, 
#                     dilation_depth, n_repeat)
#     outputs = tcn_layer(input_ts, input_static_emb, input_dynamic_emb)

#     return Model(inputs=[input_ts, input_static_emb, input_dynamic_emb], outputs=outputs)

class DeepTCN3(nn.Module):
    def __init__(self, n_back, n_fore, dilation_list, conv_filter, conv_ksize, dilation_depth, 
                n_repeat, ts_dim, static_dim, dynamic_dim):
        super(DeepTCN3, self).__init__()
        self.tcn_layer = DeepTCN3Layer(ts_dim, static_dim, dynamic_dim, n_back, n_fore, dilation_list, conv_filter, conv_ksize, 
                                    dilation_depth, n_repeat)
    
    def forward(self, input_ts, input_static_emb, input_dynamic_emb):
        return self.tcn_layer(input_ts, input_static_emb, input_dynamic_emb)
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from IPython import embed


class ResidualTCN(nn.Module):
    """Residual Temporal Convolutional Network block, as a encoder module.
    
    Args:
        - *d*: Integer, dilation rate of `Conv1D`.
        - *conv_filter*: Integer, filter size of residual `Conv1D`.
        - *k*: Integer, kernel size of `Conv1D`.

    forward Arguments:
        - Tensor, shape of `[batch_size, ]`.

    Output:
        Tensor, shape of `[batch_size, ]`.
    """
    def __init__(self, in_features, conv_filter=11, k=2, d=1, **kwargs):
        super(ResidualTCN, self).__init__(**kwargs)
        self.conv1 = nn.Conv1d(in_features, out_channels=conv_filter, kernel_size=k, dilation=d)
        self.bn1 = nn.BatchNorm1d(conv_filter)
        self.conv2 = nn.Conv1d(in_features, out_channels=conv_filter, kernel_size=k, dilation=d)
        self.bn2 = nn.BatchNorm1d(conv_filter)
            
    def forward(self, inputs):
        out = F.relu(self.bn1(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + inputs[:, :, -out.shape[2]:])


class futureResidual(nn.Module):
    """Decoder module.

    Args:
        out_features: Integer, the dimensionality of covariates embedding.

    forward Arguments:
        hidden_emb: Tensor with shape `[batch_size, n_fore, out_features]`
        input_cov_emb: Tensor with shape `[batch_size, n_fore, n_cov_emb]`

    Outputs:
        `Tensor` with shape `[batch_size, n_fore, out_features]`.
    """
    def __init__(self, in_features, out_features, **kwargs):
        super(futureResidual, self).__init__(**kwargs)
        self.fc1 = nn.Linear(in_features, 64)
        self.bn1 = nn.BatchNorm1d(24)
        self.fc2 = nn.Linear(64, out_features)
        self.bn2 = nn.BatchNorm1d(24)
        
    def forward(self, hidden_emb, input_cov_emb):
        out = F.relu(self.bn1(self.fc1(input_cov_emb)))
        out = self.bn2(self.fc2(out))
        return F.relu(hidden_emb + out) 


class Decoder(nn.Module):
    def __init__(self, in_features, out_features, hid_dim, seq_len, dropout):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_features, hid_dim)
        self.bn = nn.BatchNorm1d(seq_len)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hid_dim, out_features)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out_hid = self.dropout(self.relu1(self.bn(self.fc1(x))))
        outputs = self.relu2(self.fc2(out_hid))
        return outputs


class DeepTCN3Layer(nn.Module):
    """Temporal Convolutional Network. 

    Args:
        - *dilation_depth*: List, .
        - *n_repeat*: Integer, 

    forward Arguments:
        input_ts: `Tensor` with shape `[batch_size, n_back, 1]`.
        input_static_emb: `Tensor` with shape `[batch_size, n_fore, dim1]`.
        input_dynamic_emb: `Tensor` with shape `[batch_size, n_fore, dim2]`.

    Outputs:
        `Tensor` with shape `[batch_size, n_fore]`
    """
    def __init__(self, ts_dim, static_dim, dynamic_dim, n_back, n_fore, dilation_list=[1,2,4,8,16,20,32], 
                conv_filter=11, conv_ksize=2, dilation_depth=2, n_repeat=5, **kwargs):
        super(DeepTCN3Layer, self).__init__(**kwargs)
        self.dilation_list = dilation_list
        self.dilation_depth = dilation_depth
        self.n_repeat = n_repeat
        self.n_back = n_back
        self.n_fore = n_fore
        self.conv_filter = conv_filter
        self.conv_ksize = conv_ksize
        out_features_futureResidual = n_back
        for dilation in dilation_list:
            out_features_futureResidual -= 2 * (conv_ksize - 1) * dilation
        
        self.TCN_list = []
        for d in self.dilation_list:
            self.TCN_list.append(ResidualTCN(ts_dim+static_dim, self.conv_filter, k=self.conv_ksize, d=d))
        
        self.decoder_predict = Decoder(out_features_futureResidual*self.conv_filter, 1, 64, 24, 0.2)
        self.decoder_res = futureResidual(dynamic_dim, out_features_futureResidual*self.conv_filter)
                       
    def forward(self, input_ts, input_static_emb, input_dynamic_emb):
        # input_ts: [batch_size, n_back, 1], time series.
        # input_static_emb: [batch_size, n_back+n_fore, emb_dim].
        # input_dynamic_emb: [batch_size, n_fore, dim_total], covariate.
        # preprocess
        output = torch.cat([input_static_emb[:, :self.n_back, :], input_ts], dim=2).permute(0, 2, 1)
        for tcn_layer in self.TCN_list:
            output = tcn_layer(output)    
        output = output.permute(0, 2, 1) # output: [batch_size, n_back, conv_filter]
        output = torch.reshape(output, [output.shape[0], 1, -1])   # output: [batch_size, 1, out_features_futureResidual*conv_filter]
        output = output.repeat([1, self.n_fore, 1])
        # output: [batch_size, n_fore, 11*2], embed_concat: [batch_size, n_fore, dim_total]
        dynamic_emb = torch.cat([input_static_emb[:, -self.n_fore:, :], input_dynamic_emb], dim=2)
        output = self.decoder_predict(self.decoder_res(output, dynamic_emb))  
        return output   # output: [batch_size, n_fore, 1]
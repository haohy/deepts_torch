import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from IPython import embed


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class ResidualCNN(nn.Module):
    def __init__(self, in_features, out_features=11, k=2, d=1, **kwargs):
        super(ResidualCNN, self).__init__(**kwargs)
        self.conv1 = nn.Conv1d(in_features, out_channels=out_features, kernel_size=k, dilation=d)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.conv2 = nn.Conv1d(in_features, out_channels=out_features, kernel_size=k, dilation=d)
        self.bn2 = nn.BatchNorm1d(out_features)
            
    def forward(self, inputs):
        # inputs: [batch_size, n_back, emb_dim+1]
        # outputs: [batch_size, n_back, emb_dim+1 - 2*(k-1)]
        inputs = inputs.permute(0, 2, 1)
        out = F.relu(self.bn1(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + inputs[:, :, -out.shape[2]:]).permute(0, 2, 1)


class ResidualTCN(nn.Module):
    def __init__(self, in_features, out_features=11, k=2, d=1, dropout=0.1):
        super(ResidualTCN, self).__init__()
        layers_list = []
        layers_list.append(
            weight_norm(nn.Conv1d(in_features, out_features, k, padding=(k-1)*d, dilation=d)))
        layers_list.append(Chomp1d((k-1)*d)) 
        layers_list.append(nn.BatchNorm1d(out_features)) 
        layers_list.append(nn.ReLU())
        layers_list.append(nn.Dropout(dropout))
        layers_list.append(
            weight_norm(nn.Conv1d(out_features, out_features, k, padding=(k-1)*d, dilation=d)))
        layers_list.append(Chomp1d((k-1)*d)) 
        layers_list.append(nn.BatchNorm1d(out_features)) 
        layers_list.append(nn.ReLU())
        layers_list.append(nn.Dropout(dropout))
        self.tcn_layer = nn.Sequential(*layers_list)
        self.downsample = nn.Conv1d(in_features, out_features, 1) \
            if in_features != out_features else None
            
    def forward(self, inputs):
        # inputs: [batch_size, n_back, in_features]
        # outputs: [batch_size, n_back, out_features]
        inputs = inputs.permute(0, 2, 1)
        out = self.tcn_layer(inputs)
        res = self.downsample(inputs) if self.downsample is not None else inputs
        return F.relu(out + res).permute(0, 2, 1), None


class ResidualTCAN(nn.Module):
    def __init__(self, in_features, out_features=11, k=2, d=1, heads=2, dropout=0.1):
        super(ResidualTCAN, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_features, out_features, k, padding=(k-1)*d, dilation=d))
        self.chomp1 = Chomp1d((k-1)*d)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.multiheadattn1 = nn.MultiheadAttention(out_features, heads)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(out_features, out_features, k, padding=(k-1)*d, dilation=d))
        self.chomp2 = Chomp1d((k-1)*d)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.multiheadattn2 = nn.MultiheadAttention(out_features, heads)
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_features, out_features, 1) \
            if in_features != out_features else None
            
    def forward(self, inputs):
        # inputs: [batch_size, n_back, in_features]
        # outputs: [batch_size, n_back, out_features]
        n_back = inputs.shape[1]
        batch_size = inputs.shape[0]
        inputs = inputs.permute(0, 2, 1)
        out_tcn_1 = self.bn1(self.chomp1(self.conv1(inputs)))
        out_tcn_1 = out_tcn_1.permute(2, 0, 1)
        attn_mask = torch.tensor([[1 if i<j else 0 for j in range(n_back)] for i in range(n_back)])\
            .bool().to(inputs.device)
        out_attn_1, attn_weight_1 = self.multiheadattn1(out_tcn_1, out_tcn_1, out_tcn_1, attn_mask=attn_mask)
        out_attn_1 = F.relu(out_attn_1).permute(1, 2, 0)
        out_tcn_2 = self.bn2(self.chomp2(self.conv2(out_attn_1)))
        out_tcn_2 = out_tcn_2.permute(2, 0, 1)
        out_attn_2, attn_weight_2 = self.multiheadattn2(out_tcn_2, out_tcn_2, out_tcn_2, attn_mask=attn_mask)
        out_attn_2 = F.relu(out_attn_2).permute(1, 0, 2)
        res_en = torch.sum(attn_weight_2, -1)[...,None].repeat(1,1,inputs.shape[1]) * inputs.permute(0,2,1)
        res = self.downsample(inputs) if self.downsample is not None else inputs
        attn_weight_detach = attn_weight_2.cpu().detach().numpy()
        attn_return = [attn_weight_detach[0], attn_weight_detach[batch_size//2], attn_weight_detach[-1]]
        # return F.relu(out_attn_2 + res.permute(0, 2, 1) + res_en), attn_return
        return F.relu(out_attn_2 + res.permute(0, 2, 1)), attn_return


class futureResidual(nn.Module):
    def __init__(self, in_features, out_features, n_fore, hid_dim=64):
        super(futureResidual, self).__init__()
        self.fc1 = nn.Linear(in_features, hid_dim)
        self.bn1 = nn.BatchNorm1d(n_fore)
        self.fc2 = nn.Linear(hid_dim, out_features)
        self.bn2 = nn.BatchNorm1d(n_fore)
        
    def forward(self, hidden_emb, input_cov_emb):
        # hidden_emb: [batch_size, n_fore, out_features]
        # input_cov_emb: [batch_size, n_fore, in_features]
        out_1 = F.relu(self.bn1(self.fc1(input_cov_emb)))
        out = self.bn2(self.fc2(out_1))
        return F.relu(out + hidden_emb) 


class Decoder(nn.Module):
    def __init__(self, in_features, out_features, hid_dim, seq_len, dropout=0.2):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_features, hid_dim)
        self.bn = nn.BatchNorm1d(seq_len)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hid_dim, out_features)

    def forward(self, inputs):
        out_hid =self.relu1(self.bn(self.fc1(inputs)))
        outputs = self.fc2( self.dropout(out_hid))
        return outputs


class DeepTCN3Layer(nn.Module):
    def __init__(self, ts_dim, static_dim, dynamic_dim, n_back, n_fore, dilation_list=[1,2,4,8], 
                out_features=11, hid_dim_fore=32, conv_ksize=2, nheads=2, dilation_depth=2, n_repeat=5, **kwargs):
        super(DeepTCN3Layer, self).__init__(**kwargs)
        self.dilation_list = dilation_list
        self.dilation_depth = dilation_depth
        self.n_repeat = n_repeat
        self.n_back = n_back
        self.n_fore = n_fore
        self.out_features = out_features
        self.conv_ksize = conv_ksize
        self.hid_dim_fore = hid_dim_fore
        self.num_select = n_back
        for dilation in dilation_list:
            self.num_select -= 2 * (conv_ksize - 1) * dilation
        
        self.multi_blocks = nn.ModuleList()
        self.attentions =[]
        for d in self.dilation_list:
            # self.multi_blocks.append(ResidualTCN(out_features, out_features, k=conv_ksize, d=d))
            self.multi_blocks.append(ResidualTCAN(out_features, out_features, conv_ksize, d, nheads))
            # self.multi_blocks.append(nn.LSTM(out_features, out_features))
        
        self.upsample = nn.Linear(ts_dim, out_features)
        self.simple_resnet = nn.Linear(self.num_select*out_features, hid_dim_fore)
        self.decoder_cov = futureResidual(dynamic_dim, hid_dim_fore, n_fore)
        self.decoder_predict = Decoder(hid_dim_fore, 1, hid_dim_fore, n_fore, 0.2)

    def forward(self, input_ts, input_static_emb, input_dynamic_emb):
        # input_ts: [batch_size, n_back, 1], time series.
        # input_static_emb: [batch_size, n_back+n_fore, emb_dim].
        # input_dynamic_emb: [batch_size, n_fore, dim_total], covariate.
        batch_size = input_ts.shape[0]
        # feat_back = torch.cat([input_static_emb[:, :self.n_back, :], input_ts], dim=2)
        # feat_back = torch.cat([input_dynamic_emb[:, :self.n_back, :], input_ts], dim=2)
        feat_back = self.upsample(input_ts)
        # feat_back: [batch_size, n_back, emb_dim+1]
        for sub_layer in self.multi_blocks:
            feat_back, attn_matrixes = sub_layer(feat_back)

        feat_hid = self.simple_resnet(torch.reshape(feat_back[:,-self.num_select:,:], [batch_size, 1, -1]))
        feat_hid = feat_hid.repeat([1, self.n_fore, 1])

        # feat_fore = torch.cat([input_static_emb[:, -self.n_fore:, :], input_dynamic_emb], dim=2)
        feat_fore = input_dynamic_emb[:, -self.n_fore:, :]

        # feat_hid: [batch_size, n_fore, hid_dim_fore]
        # feat_fore: [batch_size, n_fore, dim_total]
        feat_all = self.decoder_cov(feat_hid, feat_fore)

        # feat_all: [batch_size, n_fore, hid_dim_fore]
        outputs = self.decoder_predict(feat_all)
        return torch.squeeze(outputs, dim=-1), attn_matrixes
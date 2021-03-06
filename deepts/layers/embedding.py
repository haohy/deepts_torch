import numpy as np
import torch
from torch import nn

from IPython import embed


def static_embedding(static_feat_num_dict, n_back, n_fore, period='backward', emb_dim=2):
    total = sum([v[0] for k,v in static_feat_num_dict.items()])
    if period == 'backward':
        feat_raw = np.zeros((total, n_back))
        v_idx = 0
        for k, v in static_feat_num_dict.items():
            feat_raw[v_idx: v_idx+v[0]] = np.tile([k], [v[0], n_back])
            v_idx += v[0]
    elif period == 'future':
        feat_raw = np.zeros((total, n_fore))
        v_idx = 0
        for k, v in static_feat_num_dict.items():
            feat_raw[v_idx: v_idx+v[0]] = np.tile([k], [v[0], n_fore])
            v_idx += v[0]
    elif period == 'all':
        feat_raw = np.zeros((total, n_back + n_fore))
        v_idx = 0
        for k, v in static_feat_num_dict.items():
            feat_raw[v_idx: v_idx+v[0]] = np.tile([k], [v[0], n_back + n_fore])
            v_idx += v[0]
    if emb_dim > 0:
        emb_layer = nn.Embedding(len(static_feat_num_dict.keys()), emb_dim)
        return emb_layer(torch.LongTensor(feat_raw))
    else:
        return torch.zeros(feat_raw.shape[0]*feat_raw.shape[1], len(static_feat_num_dict.keys()))\
                        .scatter_(1, feat_raw.view(-1, 1), 1)\
                            .view(feat_raw.shape[0], feat_raw.shape[1], len(static_feat_num_dict.keys()))

def dynamic_feature_cat_embedding(dynamic_feature_cat, dynamic_feat_cat_dict):
    emb_layer_list = []
    embedding_dim_all = sum([v[2] for k,v in dynamic_feat_cat_dict.items()])
    feat_embedding = torch.zeros([*dynamic_feature_cat.shape[:2], embedding_dim_all])
    
    emb_dim_idx = 0
    for i, (k, v) in enumerate(dynamic_feat_cat_dict.items()):
        input_dim = len(np.unique(dynamic_feature_cat[:, :, i]))
        emb_layer = nn.Embedding(input_dim, v[-1])
        feat_cat_dict = {}
        for j, fc in enumerate(np.unique(dynamic_feature_cat[:, :, i])):
            if  not isinstance(fc, str):
                feat_cat_dict[fc] = j
            else:
                feat_cat_dict[fc] = j
        feat_cat = np.vectorize(lambda x: feat_cat_dict[x])(dynamic_feature_cat[:, :, i])
        # try:
        #     feat_cat = dynamic_feature_cat[:, :, i].apply_(lambda x: feat_cat_dict[x])
        # except:
        if v[1] > 0:
            feat_embedding[:, :, emb_dim_idx:emb_dim_idx+v[-1]] = emb_layer(torch.LongTensor(feat_cat))
        else:
            try:
                feat_cat = torch.LongTensor(feat_cat)
                feat_embedding[:, :, emb_dim_idx:emb_dim_idx+v[-1]] = torch.zeros(
                    feat_cat.shape[0]*feat_cat.shape[1], v[-1])\
                        .scatter_(1, feat_cat.view(-1, 1), 1)\
                            .view(feat_cat.shape[0], feat_cat.shape[1], v[-1])
            except:
                embed(header="embedding")

        emb_dim_idx += v[-1]
        emb_layer_list.append(emb_layer)
    return feat_embedding
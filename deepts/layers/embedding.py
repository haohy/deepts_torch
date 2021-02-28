import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


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
        emb_layer = layers.Embedding(len(static_feat_num_dict.keys()), emb_dim)
        return emb_layer(feat_raw)
    else:
        return tf.one_hot(feat_raw, len(static_feat_num_dict.keys()))

def dynamic_feature_cat_embedding(dynamic_feature_cat, dynamic_feat_cat_dict):
    emb_layer_list = []
    embedding_dim_all = sum([v[2] for k,v in dynamic_feat_cat_dict.items()])
    feat_embedding = np.zeros([*dynamic_feature_cat.shape[:2], embedding_dim_all])
    emb_dim_idx = 0
    for i, (k, v) in enumerate(dynamic_feat_cat_dict.items()):
        input_dim = len(np.unique(dynamic_feature_cat[:, :, i]))
        emb_layer = layers.Embedding(input_dim, v[-1])
        feat_cat = dynamic_feature_cat[:, :, i] - dynamic_feature_cat[:, :, i].min()
        feat_embedding[:, :, emb_dim_idx:emb_dim_idx+v[-1]] = emb_layer(feat_cat)
        emb_dim_idx += v[-1]
        emb_layer_list.append(emb_layer)
    return feat_embedding
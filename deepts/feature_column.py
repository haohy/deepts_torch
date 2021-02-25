import numpy as np
import pandas as pd
from collections import namedtuple
import tensorflow as tf

from deepts.layers import get_embedding_layer


class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'is_future', 'is_embed', 'dimension', 'dtype'])):
    __slot__ = ()

    def __new__(cls, name, is_future, is_embed, dimension, dtype):
        
        return super(SparseFeat, cls).__new__(cls, name, is_future, is_embed, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


class DenseFeat(namedtuple('DenseFeat',
                            ['name', 'is_future', 'dimension', 'dtype'])):
    __slot__ = ()

    def __new__(cls, name, is_future, dimension=1, dtype='float32'):
        return super(DenseFeat, cls).__new__(cls, name, is_future, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


# class SparseFeat(namedtuple('SparseFeat',
#                             ['name', 'vocab_size', 'dtype', 'embed_dim'])):
#     __slot__ = ()

#     def __new__(cls, name, vocab_size, dtype, embed_dim):
        
#         return super(SparseFeat, cls).__new__(cls, name, vocab_size, dtype, embed_dim)

#     def __hash__(self):
#         return self.name.__hash__()


# class DenseFeat(namedtuple('DenseFeat',
#                             ['name', 'dimension', 'dtype'])):
#     __slot__ = ()

#     def __new__(cls, name, dimension=1, dtype='float32'):
#         return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

#     def __hash__(self):
#         return self.name.__hash__()


# class SparseFeat(namedtuple('SparseFeat',
#                             ['name', 'is_dynamic', 'is_future', 'is_embed', 'dimension', 'dtype'])):
#     __slot__ = ()

#     def __new__(cls, name, is_dynamic, is_future, is_embed, dimension, dtype):
        
#         return super(SparseFeat, cls).__new__(cls, name, is_dynamic, is_future, is_embed, dimension, dtype)

#     def __hash__(self):
#         return self.name.__hash__()


# class DenseFeat(namedtuple('DenseFeat',
#                             ['name', 'is_dynamic', 'is_future', 'dimension', 'dtype'])):
#     __slot__ = ()

#     def __new__(cls, name, is_dynamic, is_future, dimension=1, dtype='float32'):
#         return super(DenseFeat, cls).__new__(cls, name, is_dynamic, is_future, dimension, dtype)

#     def __hash__(self):
#         return self.name.__hash__()


class SequenceFeat(namedtuple('SequenceFeat',
                            ['name', 'seq_len', 'dtype'])):
    __slot__ = ()

    def __new__(cls, name, seq_len, dtype):
        return super(SequenceFeat, cls).__new__(cls, name, seq_len, dtype)

    def __hash__(self):
        return self.name.__hash__()


def get_embedding_features(X, sparse_feature_columns):
    feature_dict = {}
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    for sfc in sparse_feature_columns:
        if sfc.embed_dim > 0:
            feature_dict[sfc.name] = get_embedding_layer(sfc.vocab_size, sfc.embed_dim)(
                tf.convert_to_tensor(X[sfc.name]))
        else:
            feature_dict[sfc.name] = tf.one_hot(tf.convert_to_tensor(X[sfc.name]),
                sfc.vocab_size)
    
    return feature_dict

def get_dense_features(X, dense_feature_columns):
    feature_dict = {}
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    for dfc in dense_feature_columns:
        feature_dict[dfc.name] = tf.convert_to_tensor(X[dfc.name], dtype=dfc.dtype)[...,None]

    return feature_dict
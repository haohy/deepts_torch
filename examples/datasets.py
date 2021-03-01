import os
import numpy as np
import pandas as pd

from deepts.feature_column import SparseFeat, DenseFeat


def get_raw_df(config, ds_name, *args, **kwargs):
    """Get bike hour dataset. 
    
    :return: pd.DataFrame and pd.Series.
    """
    conf = config[ds_name]
    file_path = os.path.join(config['dir_root'], conf['file_path'])
    df_raw = pd.read_csv(file_path, **conf['csv_kwargs'])
    sparse_col = conf['sparse_col']
    dense_col = conf['dense_col']

    data_numpy_all = {}
    feature_columns = []
    for col, embed_dim in sparse_col.items():
        vocab_size = len(df_raw[col].unique())
        feature_columns.append(SparseFeat('sparse_'+col, vocab_size, 'int64', embed_dim))
        data_numpy_all['sparse_' + col] = df_raw[col].values - df_raw[col].values.min()
    for col in dense_col:
        feature_columns.append(DenseFeat('dense_' + col, 1, 'float32'))
        data_numpy_all['dense_' + col] = df_raw[col].values
    data_numpy_all['target'] = df_raw[conf['target']].values
    data_df = pd.DataFrame(data_numpy_all)
    Y = data_df.pop('target')
    X = data_df

    return X, Y, feature_columns
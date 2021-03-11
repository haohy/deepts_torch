import numpy as np
import torch


def naive_pred_error(y_true, period):
    assert len(y_true.shape) == 2, "y_true.shape.ndims != 2"
    y_t_pred = y_true[:, :-period]
    y_t_true = y_true[:, period:]
    mae_npe = torch.mean(torch.abs(y_t_true - y_t_pred))
    return mae_npe

def MASE(y_true, y_pred, sample_weight=None, period=1, epsilon=1e-10):
    assert list(y_true.shape) == list(y_pred.shape), \
        "y_true's shape {} != y_pred's shape {}.".format(y_true.shape, y_pred.shape)
    assert y_pred.shape[-1] > period, "period can't be larger than n_fore."
    mae = torch.mean(torch.abs(y_true - y_pred))
    mae_npe = naive_pred_error(y_true, period)
    return mae / (mae_npe + epsilon)

def ND(y_true, y_pred, sample_weight=None, period=1):
        denominator = torch.sum(torch.abs(y_true))
        diff = torch.sum(torch.abs(y_true - y_pred))
        return diff/denominator

def NRMSE(y_true, y_pred, sample_weight=None, period=1):
        denominator = torch.mean(y_true)
        diff = torch.sqrt(torch.mean(((y_pred - y_true)**2)))
        return diff/denominator

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers, losses, optimizers, metrics

from deepts.layers import LSTMLayer
from deepts.utils import variable_scope, root_mean_squared_error

def LSTMEstimator(model_dir, hid_dim, lr, config=None):
    
    def _model_fn(features, labels, mode, config):
        train_flag = (mode == tf.estimator.ModeKeys.TRAIN)

        with variable_scope('deepts_estimator'):
            model = LSTMLayer(hid_dim)
            logits = model(features)
            predictions = tf.squeeze(logits, axis=-1)
            loss = tf.compat.v1.losses.mean_squared_error(labels, predictions)

            # metric
            def metric_fn(labels, predictions):
                rmse, rmse_update = root_mean_squared_error(labels, predictions)
                return {'rmse': (rmse, rmse_update)}

            # evaluate
            if not train_flag:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                loss=loss,
                                                eval_metric_ops=metric_fn(labels, predictions))

            # train
            optimizer = tf.compat.v1.train.AdamOptimizer(lr)
            train_op = optimizer.minimize(loss,
                                        var_list=model.trainable_variables,
                                        global_step=tf.compat.v1.train.get_or_create_global_step()) 

        return tf.estimator.EstimatorSpec(mode=mode, 
                                        loss=loss,
                                        train_op=train_op,
                                        eval_metric_ops=metric_fn(labels, predictions))

    return tf.estimator.Estimator(_model_fn, model_dir, config)
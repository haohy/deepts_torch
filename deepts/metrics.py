import tensorflow as tf
from tensorflow.keras.metrics import Metric, MAE

class MASE(Metric):
    def __init__(self, name='mean_absolute_scaled_error', **kwargs):
        super(MASE, self).__init__(name, **kwargs)
        self.epsilon = tf.constant(1e-9, dtype=tf.float64)
        self.count = self.add_weight(name='matric_count', initializer='zeros', dtype=tf.float64)
        self.mase = self.add_weight(name='mae', initializer='zeros', dtype=tf.float64)

    def naive_pred_error(self, y_true, period):
        assert len(y_true.shape) == 2, "y_true.shape.ndims != 2"
        y_t_pred = y_true[:, :-period]
        y_t_true = y_true[:, period:]
        mae_npe =tf.cast(tf.reduce_mean(tf.abs(y_t_true - y_t_pred)), tf.float64)
        return mae_npe

    def update_state(self, y_true, y_pred, sample_weight=None, period=1):
        assert list(y_true.shape) == list(y_pred.shape), "y_true's shape != y_pred's shape."
        assert y_pred.shape[-1] > period, "period can't be larger than n_fore."
        # y_true = tf.cast(y_true, tf.float32)
        # y_pred = tf.cast(y_pred, tf.float32)
        mae = tf.cast(tf.reduce_mean(MAE(y_true, y_pred)), tf.float64)
        mae_npe = self.naive_pred_error(y_true, period)
        self.count.assign_add(1.0)
        self.mase.assign_add(mae / (mae_npe + self.epsilon))

    def result(self):
        return tf.cast(self.mase / self.count, tf.float32)

    def reset_states(self):
        self.mase.assign(0.0)


if __name__ == '__main__':
    y_true = tf.convert_to_tensor([[1,1,1,1], [1,1,1,1]], dtype='float32')
    y_pred = tf.convert_to_tensor([[1,1,1,1], [1,1,1,1]], dtype='float32')
    mase = MASE()(y_true, y_pred)
    print(mase)
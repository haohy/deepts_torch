import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers


class LSTMLayer(layers.Layer):
    """LSTM Layer.
    
    Args:
        - *units*: Integer, hidden layer's dimensionality.
        # - *n_fore*: Integer, forecast horizons.
    
    Call Arguments:
        - inputs: `FloatTensor` of shape `[batch_size, n_back, n_features]`
    
    Output shape:
        `FloatTensor` of shape `[batch_size, n_fore]`
    """
    # def __init__(self, units, n_fore=1):
    #     super(LSTMLayer, self).__init__()
    #     self.units = units
    #     self.n_fore = n_fore
    def __init__(self, units):
        super(LSTMLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer=initializers.glorot_uniform)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=initializers.glorot_uniform)
        self.bias = self.add_weight(
            shape=(self.units * 4,),
            name='bias',
            initializer=initializers.zeros)
        # self.w_out = self.add_weight(
        #     shape=(self.units, self.n_fore),
        #     name='w_output',
        #     initializer=initializers.glorot_uniform)
        # self.b_out = self.add_weight(
        #     shape=(self.n_fore, ),
        #     name='b_output',
        #     initializer=initializers.zeros)
        super(LSTMLayer, self).build(input_shape)
            
    def init_states(self):
        h = tf.random.normal((1, self.units), name='h_init')
        c = tf.random.normal((1, self.units), name='c_init')
        return (h, c)
            
    @tf.function
    def train_history(self, accumulator, x):
        h, c = accumulator
        x_ = tf.add(tf.matmul(x, self.kernel, name='x_kernel_matmul'), self.bias, name='x_kernel_add_bias')
        x_i, x_f, x_o, x_c = tf.split(x_, num_or_size_splits=4, axis=1, name='x_split_to_ifoc')
        h_ = tf.matmul(h, self.recurrent_kernel, name='h_recurrent_kernel_matmul')
        h_i, h_f, h_o, h_c = tf.split(h_, num_or_size_splits=4, axis=1, name='h_split_to_ifoc')
        i = tf.nn.sigmoid(x_i + h_i, name='i_sigmoid')
        f = tf.nn.sigmoid(x_f + h_f, name='f_sigmoid')
        o = tf.nn.sigmoid(x_o + h_o, name='o_sigmoid')
        c_ = tf.nn.tanh(x_c + h_c, name='c_tanh')
        c_t = f * c + i * c_
        h_t = o * tf.nn.tanh(c_t, name='h_i_tanh')
        return (h_t, c_t)

    # @tf.function
    # def predict_whole(self, h):
    #     y_preds = tf.add(tf.matmul(h, self.w_out), self.b_out)
    #     y_preds = tf.reshape(y_preds, shape=(self.n_fore,))
    #     return y_preds

    def step(self, input_batch, inp_batch_name):
        """Sub-step of training.

        :param input_batch: tf.FloatTensor, shape of `[n_back, n_features]`.
        :param inp_batch_name: string, name of this subblock.
        :return: tf.FloatTensor, shape of `[1, n_fore]`.
        """
        initializer = self.init_states()
        input_batch = tf.expand_dims(input_batch, axis=1, name='input_batch_expand_dims')
        h_ts, _ = tf.scan(self.train_history, input_batch, initializer, name='scan' + inp_batch_name)
        # y_preds = self.predict_whole(h_ts[-1])
        # return y_preds
        return h_ts[-1]
        
    def call(self, inputs):
        h_all = tf.TensorArray(tf.float32, size=0, dynamic_size=True, name='y_all')
        for inp_batch in inputs:
            # y_preds = self.step(inp_batch, inp_batch.name[-1]) 
            # y_all = y_all.write(y_all.size(), y_preds, name='y_all_writer')
            h_batch = self.step(inp_batch, inp_batch.name[-1]) 
            h_all = h_all.write(h_all.size(), h_batch, name='y_all_writer')
        # y_ = tf.reshape(y_all.stack(), shape=(len(inputs), self.n_fore))
        h_ = tf.reshape(h_all.stack(), shape=(len(inputs), self.units))
        return h_

    def get_config(self):
        config = {
            'units': self.units
        }
        base_config = self.get_config()
        return base_config.update(config)


class TCAN(layers.Layer):
    """"""
    def __init__(self, args):
        pass

    def build(self, input_shape):
        pass

    def call(self, inputs):
        pass

    def get_config(self):
        config = {
            'ele': self.ele
        }
        base_config = self.get_config()
        return base_config.update(config)


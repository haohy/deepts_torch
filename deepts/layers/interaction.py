import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras import initializers
from tensorflow.python.ops import array_ops, init_ops, math_ops, nn, control_flow_util
from tensorflow.python.keras import backend as K


def _lower_triangular_mask(shape):
    """Creates a lower-triangular boolean mask over the last 2 dimensions."""
    row_index = math_ops.cumsum(
        array_ops.ones(shape=shape, dtype=tf.int32), axis=-2)
    col_index = math_ops.cumsum(
        array_ops.ones(shape=shape, dtype=tf.int32), axis=-1)
    return math_ops.greater_equal(row_index, col_index)

def _merge_masks(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return math_ops.logical_and(x, y)

class AttentionBlock(layers.Attention):
    """Dot-product attention layer, a.k.a. Luong-style attention.
    
    Inputs are `query` tensor of shape `[batch_size, Tq, dim]`, `value` tensor of
    shape `[batch_size, Tv, dim]` and `key` tensor of shape
    `[batch_size, Tv, dim]`. The calculation follows the steps:
    
    1. Calculate scores with shape `[batch_size, Tq, Tv]` as a `query`-`key` dot
        product: `scores = tf.matmul(query, key, transpose_b=True)`.
    2. Use scores to calculate a distribution with shape
        `[batch_size, Tq, Tv]`: `distribution = tf.nn.softmax(scores)`.
    3. Use `distribution` to create a linear combination of `value` with
        shape `[batch_size, Tq, dim]`:
        `return tf.matmul(distribution, value)`.
    
    Args:
        use_scale: If `True`, will create a scalar variable to scale the attention
        scores.
        causal: Boolean. Set to `True` for decoder self-attention. Adds a mask such
        that position `i` cannot attend to positions `j > i`. This prevents the
        flow of information from the future towards the past.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
        attention scores.
    
    Call Arguments:
        inputs: List of the following tensors:
            * query: Query `Tensor` of shape `[batch_size, Tq, dim]`.
            * value: Value `Tensor` of shape `[batch_size, Tv, dim]`.
            * key: Optional key `Tensor` of shape `[batch_size, Tv, dim]`. If not
                given, will use `value` for both `key` and `value`, which is the
                most common case.
        mask: List of the following tensors:
            * query_mask: A boolean mask `Tensor` of shape `[batch_size, Tq]`.
                If given, the output will be zero at the positions where
                `mask==False`.
            * value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`.
                If given, will apply the mask such that values at positions where
                `mask==False` do not contribute to the result.
        return_attention_scores: bool, it `True`, returns the attention scores
            (after masking and softmax) as an additional output argument.
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).
    
    Output:
        Attention outputs of shape `[batch_size, Tq, dim]`.
        [Optional] Attention scores after masking and softmax with shape
            `[batch_size, Tq, Tv]`.
    """
    def __init__(self, use_scale=False, softmax_axis=1, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.use_scale = use_scale
        self.softmax_axis = softmax_axis

    def build(self, input_shape):
        """Creates scale variable if use_scale==True."""
        if self.use_scale:
            self.scale = self.add_weight(
                name='scale',
                shape=(),
                initializer=init_ops.ones_initializer(),
                dtype=self.dtype,
                trainable=True)
        else:
            self.scale = None
        super(AttentionBlock, self).build(input_shape)

    def _apply_scores(self, scores, value, scores_mask=None, training=None):
        if scores_mask is not None:
            padding_mask = math_ops.logical_not(scores_mask)
            # Bias so padding positions do not contribute to attention distribution.
            scores -= 1.e9 * math_ops.cast(padding_mask, dtype=K.floatx())
        if training is None:
            training = K.learning_phase()
        weights = nn.softmax(scores, axis=self.softmax_axis)

        def dropped_weights():
            return nn.dropout(weights, rate=self.dropout)

        weights = tf.cond(training, dropped_weights, lambda: array_ops.identity(weights))
        return math_ops.matmul(weights, value), weights

    def _calculate_scores(self, query, key):
        """Calculates attention scores as a query-key dot product.
        Args:
            query: Query tensor of shape `[batch_size, Tq, dim]`.
            key: Key tensor of shape `[batch_size, Tv, dim]`.
        Returns:
            Tensor of shape `[batch_size, Tq, Tv]`.
        """
        scores = math_ops.matmul(query, key, transpose_b=True)
        if self.scale is not None:
            scores *= self.scale
        return scores

    def call(self,
           inputs,
           mask=None,
           training=None,
           return_attention_scores=False):
        self._validate_call_args(inputs=inputs, mask=mask)
        q = inputs[0]
        v = inputs[1]
        k = inputs[2] if len(inputs) > 2 else v
        q_mask = mask[0] if mask else None
        v_mask = mask[1] if mask else None
        scores = self._calculate_scores(query=q, key=k)
        if v_mask is not None:
            # Mask of shape [batch_size, 1, Tv].
            v_mask = array_ops.expand_dims(v_mask, axis=-2)
        if self.causal:
            # Creates a lower triangular mask, so position i cannot attend to
            # positions j>i. This prevents the flow of information from the future
            # into the past.
            scores_shape = array_ops.shape(scores)
            # causal_mask_shape = [1, Tq, Tv].
            causal_mask_shape = array_ops.concat(
                [array_ops.ones_like(scores_shape[:-2]), scores_shape[-2:]],
                axis=0)
            causal_mask = _lower_triangular_mask(causal_mask_shape)
        else:
            causal_mask = None
        scores_mask = _merge_masks(v_mask, causal_mask)
        result, attention_scores = self._apply_scores(
            scores=scores, value=v, scores_mask=scores_mask, training=training)
        if q_mask is not None:
        # Mask of shape [batch_size, Tq, 1].
            q_mask = array_ops.expand_dims(q_mask, axis=-1)
            result *= math_ops.cast(q_mask, dtype=result.dtype)
        if return_attention_scores:
            return result, attention_scores
        return result

    def get_config(self):
        config = {'use_scale': self.use_scale}
        base_config = super(AttentionBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Dense4Attention(layers.Layer):
    def __init__(self, dim):
        self.dense_layer = layers.Dense(3 * dim)
        super(Dense4Attention, self).__init__()

    def call(self, inputs):
        outputs = self.dense_layer(inputs)
        query, key, value = tf.split(outputs, 3, axis=-1)
        return query, key, value


class TemporalBlock(layers.Layer):
    """Temporal Block for TemporalConvBlock.

    Args:
        - *softmax_axis*: Integer, which axis the `softmax` function calculate along.
        - *n_inputs*: Integer, the number of convolutional layer's input channel. 
        - *n_outputs*: Integer, the number of convolutional layer's output channel. 
        - *kernel_size*: Integer, the size of convolutional layer's kernel. 
        - *num_sub_blocks*: Integer, the number of convolutional layer. 
        - *temp_attn*: Boolean, if use `Temporal Attention`.
        - *en_res*: Boolean, if use `Enhanced Residual`.
        - *is_conv*: Boolean, if use `Convolutional layers`.
        - *stride*: Integer, stride of convolutional layer.
        - *dilation*: Integer, dilation rate of convolutional layer.
        - *visual*: Boolean, if visual or not.
        - *dropout*: Float, dropout rate.

    Call Arguments:
        - *inputs*: `Tensor` of shape `[batch_size, n_back, n_inputs]`.

    Outputs:
        - Tensor of shape `[batch_size, n_back, n_outputs]`
        - [Optional] attn_weight_cpu
    """
    def __init__(self, softmax_axis, n_inputs, n_outputs, kernel_size, num_sub_blocks, attn_dim, 
                temp_attn, en_res, is_conv, stride, dilation, visual, dropout=0.2):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.kernel_size = kernel_size
        self.attn_dim = attn_dim
        self.dilation = dilation
        self.stride = stride
        self.dropout = dropout
        self.softmax_axis = softmax_axis
        self.en_res = en_res
        self.is_conv = is_conv
        self.temp_attn = temp_attn
        self.visual = visual
        self.num_sub_blocks = num_sub_blocks
        super(TemporalBlock, self).__init__()

    def build(self, input_shape):
        self.dense4attention = Dense4Attention(self.attn_dim)
        self.attention = AttentionBlock(True, self.softmax_axis)
        self.downsample = layers.Dense(self.n_outputs, use_bias=False)
        self.relu = layers.ReLU()
        if self.is_conv:
            self.net = self._make_layers(self.num_sub_blocks, self.n_outputs, self.kernel_size, 
                                        self.stride, self.dilation, self.dropout)
        else:
            self.net = layers.Dense(self.n_outputs, use_bias=False)

    def _make_layers(self, num_sub_blocks, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        layer_sequence = Sequential()
        for _ in range(num_sub_blocks):
            layer_sequence.add(layers.Conv1D(n_outputs, kernel_size, strides=stride, padding='same', 
                            dilation_rate=dilation))
            layer_sequence.add(layers.ReLU())
            layer_sequence.add(layers.Dropout(dropout))

        return layer_sequence

    def call(self, inputs):
        # inputs: [batch_size, n_back, n_inputs]
        if self.temp_attn:
            en_res_x = None
            out_attn, attn_weight = self.attention(list(self.dense4attention(inputs)), 
                                                return_attention_scores=True)
            out = self.net(out_attn)  # out: [batch_size, n_back, n_outputs]
            weight_x = tf.nn.softmax(tf.reduce_sum(attn_weight, axis=2), axis=1)   # weight_x: [batch_size, n_back]
            # weight_x_: [batch_size, n_back, n_outputs]
            weight_x_ = tf.tile(tf.expand_dims(weight_x, axis=2), [1, 1, inputs.shape[2]])
            en_res_x = tf.multiply(weight_x_, inputs)
            if en_res_x.shape[-1] != self.n_outputs:
                en_res_x = self.downsample(en_res_x)
                res = self.downsample(inputs)
            else:
                en_res_x = en_res_x
                res = inputs

            if self.visual:
                attn_weight_cpu = attn_weight.numpy()
            else:
                attn_weight_cpu = [0]*10
            del attn_weight
            
            if self.en_res:
                return self.relu(out + res + en_res_x), attn_weight_cpu
            else:
                return self.relu(out + res), attn_weight_cpu

        else:
            out = self.net(inputs)  # out: [batch_size, n_back, n_outputs]
            res = inputs if inputs.shape[-1] == self.n_outputs else self.downsample(inputs)
            return self.relu(out + res) # return: [batch_size, n_back, n_outputs]


class TemporalConvNet(layers.Layer):
    """Temporal Block for TemporalConvBlock.

    Args:
        - *emb_dim*: Integer, dimensionality of embedding.
        - *attn_dim*: Integer, dimensionality of `AttetionBlock`'s dense layer.
        - *channel_list*: List, list of multi-block's channel size.
        - *num_sub_blocks*: Integer, the number of convolutional layer. 
        - *temp_attn*: Boolean, if use `Temporal Attention`.
        - *en_res*: Boolean, if use `Enhanced Residual`.
        - *is_conv*: Boolean, if use `Convolutional layers`.
        - *softmax_axis*: Integer, which axis the `softmax` function calculate along.
        - *kernel_size*: Integer, the size of convolutional layer's kernel. 
        - *visual*: Boolean, if visual or not.
        - *dropout*: Float, dropout rate.

    Call Arguments:
        - *inputs*: `Tensor` of shape `[batch_size, n_back, emb_dim]`.

    Outputs:
        - Tensor of shape `[batch_size, n_back, n_outputs]`
        - [Optional] attn_weight_cpu
    """
    def __init__(self, emb_dim, attn_dim, channel_list, num_sub_blocks, temp_attn, en_res,
                is_conv, softmax_axis, kernel_size, visual, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.temp_attn = temp_attn
        layer_list = []
        for i in range(len(channel_list)):
            dilation_rate = 2 ** i
            in_channels = emb_dim if i == 0 else channel_list[i-1]
            out_channels = channel_list[i]
            layer_list.append(TemporalBlock(softmax_axis, in_channels, out_channels, kernel_size, 
                            num_sub_blocks, attn_dim, temp_attn, en_res, is_conv, 1, dilation_rate, 
                            visual, dropout))
        self.network = layer_list

    def call(self, inputs):
        attn_weight_list = []
        out = inputs
        if self.temp_attn:
            for i in range(len(self.network)):
                out, attn_weight = self.network[i](out)
                attn_weight_list.append([attn_weight[0], attn_weight[-1]])
            return out, attn_weight_list
        else:
            for i in range(len(self.network)):
                out = self.network[i](out)
            return out, None


class LSTM2Layer(layers.Layer):
    """LSTM Layer.
    
    Args:
        - *units*: Integer, hidden layer's dimensionality.
        # - *n_fore*: Integer, forecast horizons.
    
    Call Arguments:
        - inputs: `FloatTensor` of shape `[batch_size, n_back, n_features]`
    
    Output shape:
        `FloatTensor` of shape `[batch_size, n_fore]`
    """
    def __init__(self, units):
        super(LSTM2Layer, self).__init__()
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
        super(LSTM2Layer, self).build(input_shape)
            
    def init_states(self):
        h = tf.random.normal((1, self.units), name='h_init')
        c = tf.random.normal((1, self.units), name='c_init')
        return h, c
    
    @tf.function
    def lstm_unit(self, x, h, c):
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
        return h_t, c_t

    @tf.function
    def lstm_loop(self, inp_batch, hid_all_ret=False):
        inp_batch = tf.expand_dims(inp_batch, axis=1)
        num_loop = len(inp_batch)
        h, c = self.init_states()
        h_batch = []

        lstm_loop_start = time.time()
        for i in range(num_loop):

            # per_lstm_loop_start = time.time()
            h, c = self.lstm_unit(inp_batch[i], h, c)
            h_batch.append(h)
            # tf.print("per lstm loop cost: {}".format(time.time() - per_lstm_loop_start))

        tf.print("lstm loop cost: {}".format(time.time() - lstm_loop_start))
        if hid_all_ret:
            return h_batch
        else:
            return h
    
    # @tf.function
    def call(self, inputs):
        call_for_start = time.time()
        h_all = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for inp_batch in inputs:

            loop_start_time = time.time()
            h_ = self.lstm_loop(inp_batch)
            loop_cost = time.time() - loop_start_time
            tf.print('loop_cost: {}'.format(loop_cost))

            TensorArray_start = time.time()
            h_all = h_all.write(h_all.size(), h_)
            tf.print("TensorArray cost: {}".format(time.time() - TensorArray_start))

        tf.print("call for: {}".format(time.time() - call_for_start))

        stack_start = time.time()
        h_all = tf.reshape(h_all.stack(), shape=(len(inputs), self.units))
        tf.print("stack cost: {}".format(time.time() - stack_start))

        return h_all

    def get_config(self):
        config = {
            'units': self.units
        }
        base_config = self.get_config()
        return base_config.update(config)


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

    def step(self, input_batch, inp_batch_name):
        """Sub-step of training.

        :param input_batch: tf.FloatTensor, shape of `[n_back, n_features]`.
        :param inp_batch_name: string, name of this subblock.
        :return: tf.FloatTensor, shape of `[1, n_fore]`.
        """
        initializer = self.init_states()
        input_batch = tf.expand_dims(input_batch, axis=1, name='input_batch_expand_dims')
        h_ts, _ = tf.scan(self.train_history, input_batch, initializer, name='scan' + inp_batch_name)
        return h_ts[-1]
        
    def call(self, inputs):
        h_all = tf.TensorArray(tf.float32, size=0, dynamic_size=True, name='y_all')
        for inp_batch in inputs:
            h_batch = self.step(inp_batch, inp_batch.name[-1]) 
            h_all = h_all.write(h_all.size(), h_batch, name='y_all_writer')
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


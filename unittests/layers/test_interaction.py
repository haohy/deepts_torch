import pytest
import os, sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(1, os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from deepts.layers import AttentionBlock, TemporalBlock, TemporalConvNet
from unittests.config import SAMPLE_SIZE, BATCH_SIZE, N_BACK


@pytest.mark.parametrize(
    'use_scale, softmax_axis',
    [(True, 1), (True, 2)]
)
def test_AttentionBlock(use_scale, softmax_axis):
    Tq, Tv, dim = 10, 20, 15
    attention_block_layer = AttentionBlock(use_scale, softmax_axis)
    query = tf.random.normal((BATCH_SIZE, Tq, dim))
    key = tf.random.normal((BATCH_SIZE, Tv, dim))
    value = tf.random.normal((BATCH_SIZE, Tv, dim))
    outputs = attention_block_layer([query, key, value])
    print(outputs.shape)

@pytest.mark.parametrize(
    'temp_attn, en_res, is_conv, softmax_axis',
    [(True, True, True, 1),
    (True, True, True, 2),
    (False, True, True, 1),
    (False, False, True, 1),
    (True, False, True, 1),
    (True, True, False, 1),]
)
def test_TemporalBlock(temp_attn, en_res, is_conv, softmax_axis):
    n_inputs = 10
    n_outputs = 20
    kernel_size = 3
    num_sub_blocks = 2
    attn_dim = 10
    stride = 1
    dilation = 2
    visual = False
    temporalblock_layer = TemporalBlock(softmax_axis, n_inputs, n_outputs, kernel_size, 
                                        num_sub_blocks, attn_dim, temp_attn, en_res, is_conv, 
                                        stride, dilation, visual, dropout=0.2)
    inputs = tf.random.normal((BATCH_SIZE, N_BACK, n_inputs))
    outputs = temporalblock_layer(inputs)
    print(outputs[0].shape)

@pytest.mark.parametrize(
    'temp_attn, en_res, is_conv, softmax_axis',
    [(True, True, True, 1),
    (True, True, True, 2),
    (False, True, True, 1),
    (False, False, True, 1),
    (True, False, True, 1),
    (True, True, False, 1),]
)
def test_TemporalConvNet(temp_attn, en_res, is_conv, softmax_axis):
    emb_dim = 10
    channel_list = [20, 30, 40]
    kernel_size = 3
    num_sub_blocks = 2
    attn_dim = 10
    visual = False
    temporalconvnet_layer = TemporalConvNet(emb_dim, attn_dim, channel_list, num_sub_blocks, 
                                        temp_attn, en_res, is_conv, softmax_axis, kernel_size,
                                        visual, dropout=0.2)
    inputs = tf.random.normal((BATCH_SIZE, N_BACK, emb_dim))
    outputs = temporalconvnet_layer(inputs)
    print(outputs[0].shape)



if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.set_visible_devices(physical_devices[1:], 'GPU')
    # test_AttentionBlock(True, 2)
    # pytest.main(['-s', '-k', 'AttentionBlock', './unittests/layers/test_interaction.py'])
    # test_TemporalBlock(True, True, True, 1)
    # pytest.main(['-s', '-k', 'TemporalBlock', './unittests/layers/test_interaction.py'])
    # test_TemporalConvNet(True, True, False, 1)
    pytest.main(['-s', '-k', 'TemporalConvNet', './unittests/layers/test_interaction.py'])
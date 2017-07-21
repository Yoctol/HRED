import tensorflow as tf

from .layers import *
from .tool import check_layer_format


def query_level_encoder(layer, input_with_embedding, mask,
                        batch_size, scope_name='query_level_encoder'):
    check_layer_format(layer)
    layer_func = globals()[layer['name']]
    layer['params']['batch_size'] = batch_size
    scope_name = scope_name + '_' + layer['name']

    with tf.variable_scope(scope_name):
        cells, init_state = layer_func(**layer['params'])
        hidden_output, hidden_state = tf.nn.dynamic_rnn(
            cell=cells,
            inputs=input_with_embedding,
            sequence_length=mask,
            initial_state=init_state,
            dtype=tf.float32,
        )
    return hidden_output, hidden_state

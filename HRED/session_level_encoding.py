import tensorflow as tf

from .layers import *
from .tool import check_layer_format


def session_level_encoder(layer, query_output, query_mask,
                          scope_name='session_level_encoder'):
    check_layer_format(layer)
    layer_func = globals()[layer['name']]
    scope_name = scope_name + '_' + layer['name']

    num_querys = tf.count_nonzero([query_mask], axis=1)

    with tf.variable_scope(scope_name):
        cells, init_state = layer_func(**layer['params'])
        hidden_output, hidden_state = tf.nn.dynamic_rnn(
            cell=cells,
            inputs=query_output,
            sequence_length=num_querys,
            initial_state=init_state,
            dtype=tf.float32,
        )
    return hidden_output, hidden_state

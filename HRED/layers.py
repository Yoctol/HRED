import tensorflow as tf


def multi_grus(batch_size, state_size,
               keep_prob, num_layers,
               init_state=None):

    def basic_gru():
        gru_cell = tf.contrib.rnn.GRUCell(
            state_size,
            activation=None,
            reuse=None,
            kernel_initializer=None,
            bias_initializer=None
        )
        if keep_prob < 1:
            gru_cell = tf.contrib.rnn.DropoutWrapper(
                gru_cell, output_keep_prob=keep_prob)
        return gru_cell

    cell_grus = tf.contrib.rnn.MultiRNNCell(
        [basic_gru() for _ in range(num_layers)], state_is_tuple=True)

    if init_state is None:
        init_state = cell_grus.zero_state(batch_size, tf.float32)

    return cell_grus, init_state

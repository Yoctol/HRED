import tensorflow as tf

from .tool import cast_to_float32


def from_query_to_session_level(query_output, mask):
    [query_output] = cast_to_float32([query_output])
    query_output_shape = tf.shape(query_output, )
    zero_one_embedding = tf.concat([tf.zeros([1, query_output_shape[2]]),
                                    tf.ones([1, query_output_shape[2]])], axis=0)
    last_word_idx = tf.one_hot(
        indices=mask - 1, depth=query_output_shape[1], dtype=tf.int32)
    last_word_idx_with_embedding = tf.nn.embedding_lookup(
        zero_one_embedding, last_word_idx)
    return tf.reduce_sum(query_output * last_word_idx_with_embedding, axis=1)

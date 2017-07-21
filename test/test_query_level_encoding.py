import tensorflow as tf

from HRED.query_level_encoding import *


class QueryLevelEncodingTest(tf.test.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.layer = {
            'name': 'multi_grus',
            'params': {'state_size': 50, 'keep_prob': 0.9, 'num_layers': 3}}
        self.input_with_embedding = tf.constant(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]],
             [[4, 1, 5], [5, 1, 6], [0, 0, 0], [0, 0, 0]]])
        self.input_with_embedding = tf.cast(
            self.input_with_embedding, tf.float32)
        self.mask = tf.constant([3, 2])
        self.batch_size = 2

    def test_query_level_encoder(self):
        with self.test_session() as sess:
            output, state = query_level_encoder(
                self.layer, self.input_with_embedding, self.mask, self.batch_size)
            sess.run(tf.global_variables_initializer())
            self.assertEqual(
                (self.batch_size, 4, self.layer['params']['state_size']),
                output.eval().shape)
            self.assertEqual(self.layer['params']['num_layers'], len(state))
            self.assertEqual(
                (self.batch_size, self.layer['params']['state_size']),
                state[0].eval().shape)

import tensorflow as tf

from HRED.session_level_encoding import *


class SessionLevelEncodingTest(tf.test.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.layer = {
            'name': 'multi_grus',
            'params': {'state_size': 50, 'keep_prob': 0.9,
                       'num_layers': 3, 'batch_size': 1}}
        self.input_with_embedding = tf.constant(
            [[[1, 2, 3, 4, 5], [4, 5, 6, 7, 8], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
        self.input_with_embedding = tf.cast(
            self.input_with_embedding, tf.float32)
        self.mask = tf.constant([2, 3, 0, 0])

    def test_session_level_encoder(self):
        with self.test_session() as sess:
            output, state = session_level_encoder(
                self.layer, self.input_with_embedding, self.mask)
            sess.run(tf.global_variables_initializer())
            self.assertEqual(
                (1, 4, self.layer['params']['state_size']),
                output.eval().shape)
            self.assertEqual(self.layer['params']['num_layers'], len(state))
            self.assertEqual(
                (1, self.layer['params']['state_size']),
                state[0].eval().shape)

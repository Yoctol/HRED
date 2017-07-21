import numpy as np
import tensorflow as tf

from HRED.bridge import *


class BridgeTest(tf.test.TestCase):

    def test_from_query_to_session_level(self):
        with self.test_session() as sess:
            input_with_embedding = tf.constant(
                [[[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0]],
                 [[5, 1, 3, 4], [0, 0, 0, 0], [0, 0, 0, 0]]])
            mask = tf.constant([2, 1])
            result = from_query_to_session_level(input_with_embedding, mask)
            answer = np.array([[5., 6., 7., 8.], [5., 1., 3., 4.]])
            self.assertShapeEqual(answer, result)
            self.assertAllEqual(answer, result.eval())

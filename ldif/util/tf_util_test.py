# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""Tests for ldif.util.tf_util."""

import numpy as np
from parameterized import parameterized
import tensorflow as tf


# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import tf_util
# pylint: enable=g-bad-import-order

DISTANCE_EPS = 1e-6


class TfUtilTest(tf.test.TestCase):

  @parameterized.expand([('RemoveSecondRow', 1, 0, [[1.0, 2.0]]),
                         ('RemoveFirstRow', 0, 0, [[3.0, 4.0]]),
                         ('RemoveFirstCol', 0, 1, [[2.0], [4.0]]),
                         ('RemoveSecondCol', 1, 1, [[1.0], [3.0]])])
  def testRemoveElement(self, name, elt, axis, expected):
    initial = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    removed = tf_util.remove_element(initial, tf.constant([elt],
                                                          dtype=tf.int32), axis)
    expected = np.array(expected, dtype=np.float32)
    with self.test_session() as sess:
      returned = sess.run(removed)
    distance = float(np.sum(np.abs(expected - returned)))
    self.assertLess(
        distance, DISTANCE_EPS, 'Expected \n%s\n but got \n%s' %
        (np.array_str(expected), np.array_str(returned)))


if __name__ == '__main__':
  tf.test.main()

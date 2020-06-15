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
"""Tests for ldif.util.math_util."""

import numpy as np
from parameterized import parameterized
import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import math_util
# pylint: enable=g-bad-import-order


FLOAT_DISTANCE_EPS = 1e-5


class MathUtilTest(tf.test.TestCase):

  @parameterized.expand([('no_flatten', False),
                         ('flatten', True)])
  def test_increase_frequency(self, name, flatten):
    tf.disable_eager_execution()
    x = np.array([1.0, 0.1123, 0.7463], dtype=np.float32)
    dim = 3
    # pylint:disable=bad-whitespace
    expected = [
        [0,        -1,        0,        1,         0,          1       ],
        [0.345528,  0.938409, 0.648492, 0.761221,  0.987292,   0.158916],
        [0.715278, -0.69884, -0.99973, -0.0232457, 0.0464788, -0.998919]
    ]
    # pylint:enable=bad-whitespace
    expected = np.array(expected, dtype=np.float32)
    if flatten:
      expected = np.reshape(expected, [-1])
    y = math_util.increase_frequency(
        tf.constant(x, dtype=tf.float32), dim, flatten=flatten, interleave=True)
    with self.session() as sess:
      out = sess.run(y)
    distance = float(np.sum(np.abs(expected - out)))
    self.assertLess(distance, FLOAT_DISTANCE_EPS,
                    f'Expected {expected} but got {out}')

  @parameterized.expand([('no_flatten', False),
                         ('flatten', True)])
  def test_increase_frequency_no_interleave(self, name, flatten):
    tf.disable_eager_execution()
    x = np.array([1.0, 0.1123, 0.7463], dtype=np.float32)
    dim = 3
    # pylint:disable=bad-whitespace
    expected = [
        [0,         0,        0,         -1,        1,          1],
        [0.345528,  0.648492, 0.987292,   0.938409, 0.761221,   0.158916],
        [0.715278, -0.99973,  0.0464788, -0.69884, -0.0232457, -0.998919],
    ]
    # pylint:enable=bad-whitespace
    expected = np.array(expected, dtype=np.float32)
    if flatten:
      expected = np.reshape(expected, [-1])
    y = math_util.increase_frequency(
        tf.constant(x, dtype=tf.float32),
        dim,
        flatten=flatten,
        interleave=False)
    with self.session() as sess:
      out = sess.run(y)
    distance = float(np.sum(np.abs(expected - out)))
    self.assertLess(distance, FLOAT_DISTANCE_EPS,
                    f'Expected {expected} but got {out}')


if __name__ == '__main__':
  tf.test.main()

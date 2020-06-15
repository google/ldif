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
"""Generic Tensorflow utility functions."""

import numpy as np
import tensorflow as tf


def assert_shape(tensor, shape, name):
  """Asserts that the target and actual shape match."""
  real_shape = tensor.get_shape().as_list()
  same_rank = len(real_shape) == len(shape)
  all_equal = all([(s == r or s == -1) for s, r in zip(shape, real_shape)])
  if not same_rank or not all_equal:
    raise tf.errors.InvalidArgumentError(
        'Error: Expected tensor %s to have shape %s, but it had shape %s.' %
        (name, str(shape), str(real_shape)))


def log(msg, t):
  """Prints the tensor t with preceding message msg.

  Usage: t = tf_util.print_tf("t is: ", t)

  Args:
    msg: A string. The message that is shown before the tensor.
    t: A tensor to print.

  Returns:
    A tensor with a control dependency on the print op and a value of t.
  """
  print_op = tf.print(msg, t)
  with tf.control_dependencies([print_op]):
    t = tf.identity(t)
  return t


def tile_new_axis(t, axis, length):
  """Creates a new tensor axis and tiles it to a specified length.

  Args:
    t: Tensor with any shape.
    axis: The index for the new axis.
    length: The length of the new axis.

  Returns:
    Tensor with one extra dimension of length 'length' added to 't' at index
    'axis'.
  """
  t = tf.expand_dims(t, axis=axis)
  cur_shape = t.get_shape().as_list()
  tile_shape = [1] * len(cur_shape)
  tile_shape[axis] = length
  return tf.tile(t, tile_shape)


def zero_by_mask(mask, vals, replace_with=0.0):
  """"Sets the invalid part of vals to the value of replace_with.

  Args:
    mask: Boolean tensor with shape [..., 1].
    vals: Tensor with shape [..., channel_count].
    replace_with: Value to put in invalid locations, if not 0.0. Dtype should be
      compatible with that of vals. Can be a scalar tensor.

  Returns:
    Tensor with shape [..., channel_count].
  """
  assert mask.dtype == tf.as_dtype(np.bool)
  ms = mask.get_shape().as_list()
  vs = vals.get_shape().as_list()
  mask = tf.ensure_shape(mask, vs[:-1] + [1])
  vals = tf.ensure_shape(vals, ms[:-1] + [vs[-1]])
  vals = tf.where_v2(mask, vals, replace_with)
  return vals


def remove_element(t, elt, axis):
  """Removes the elt-th element from the axis-th axis of a tensor t.

  For example, remove_element([[1, 2], [3, 4]], Tensor(0), 0) -> [[3, 4]].

  Args:
    t: Tensor with at least rank 1.
    elt: Scalar tensor with type tf.int32.
    axis: Int pointing to one of the axes of t.

  Returns:
    Tensor with shape similar to t, except that the axis-th dimension of the
    output has one element removed.
  """
  elt = tf.cast(elt, dtype=tf.int32)
  t_shape = t.get_shape().as_list()
  elt_tensor = tf.constant(list(range(t.shape[axis].value)), dtype=tf.int32)
  num_dims_before_axis = axis
  num_dims_after_axis = len(t.shape) - axis - 1
  elt_tensor = tf.logical_not(tf.equal(elt_tensor, tf.reshape(elt, [1])))
  elt_tensor = tf.reshape(elt_tensor, [1] * num_dims_before_axis +
                          [t.shape[axis].value] + [1] * num_dims_after_axis)
  elt_tensor = tf.tile(
      elt_tensor,
      t_shape[:num_dims_before_axis] + [1] + t_shape[num_dims_before_axis + 1:])
  masked = tf.boolean_mask(t, elt_tensor)
  return tf.reshape(
      masked, t_shape[:num_dims_before_axis] + [t.shape[axis].value - 1] +
      t_shape[num_dims_before_axis + 1:])

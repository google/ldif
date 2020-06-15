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
"""Utilities for randomly sampling in tensorflow."""

import importlib
import math

import numpy as np
import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import geom_util
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order

importlib.reload(geom_util)


def random_shuffle_along_dim(tensor, dim):
  """Randomly shuffles the elements of 'tensor' along axis with index 'dim'."""
  if dim == 0:
    return tf.random_shuffle(tensor)
  tensor_rank = len(tensor.get_shape().as_list())
  perm = list(range(tensor_rank))
  perm[dim], perm[0] = perm[0], perm[dim]
  tensor = tf.transpose(tensor, perm=perm)
  tensor = tf.random_shuffle(tensor)
  tensor = tf.transpose(tensor, perm=perm)
  return tensor


def random_pan_rotations(batch_size):
  """Generates random 4x4 panning rotation matrices."""
  theta = tf.random.uniform(shape=[batch_size], minval=0, maxval=2.0 * math.pi)
  z = tf.zeros_like(theta)
  o = tf.ones_like(theta)
  ct = tf.math.cos(theta)
  st = tf.math.sin(theta)
  m = tf.stack([ct, z, st, z, z, o, z, z, -st, z, ct, z, z, z, z, o], axis=-1)
  return tf.reshape(m, [batch_size, 4, 4])


def random_pan_rotation_np():
  theta = np.random.uniform(0, 2.0 * np.pi)
  m = np.array([[np.cos(theta), 0, np.sin(theta), 0], [0, 1, 0, 0],
                [-np.sin(theta), 0, np.cos(theta), 0], [0, 0, 0, 1]],
               dtype=np.float32)
  return m


def random_rotations(batch_size):
  """Generates uniformly random 3x3 rotation matrices."""
  theta = tf.random.uniform(shape=[batch_size], minval=0, maxval=2.0 * math.pi)
  phi = tf.random.uniform(shape=[batch_size], minval=0, maxval=2.0 * math.pi)
  z = tf.random.uniform(shape=[batch_size], minval=0, maxval=2.0)

  r = tf.sqrt(z + 1e-8)
  v = tf.stack([r * tf.sin(phi), r * tf.cos(phi),
                tf.sqrt(2.0 - z + 1e-8)],
               axis=-1)
  st = tf.sin(theta)
  ct = tf.cos(theta)
  zero = tf.zeros_like(st)
  one = tf.ones_like(st)
  base_rot = tf.stack([ct, st, zero, -st, ct, zero, zero, zero, one], axis=-1)
  base_rot = tf.reshape(base_rot, [batch_size, 3, 3])
  v_outer = tf.matmul(v[:, :, tf.newaxis], v[:, tf.newaxis, :])
  rotation_3x3 = tf.matmul(v_outer - tf.eye(3, batch_shape=[batch_size]),
                           base_rot)
  return rotation_to_tx(rotation_3x3)


def random_rotation_np():
  """Returns a uniformly random SO(3) rotation as a [3,3] numpy array."""
  vals = np.random.uniform(size=(3,))
  theta = vals[0] * 2.0 * np.pi
  phi = vals[1] * 2.0 * np.pi
  z = 2.0 * vals[2]
  r = np.sqrt(z)
  v = np.stack([r * np.sin(phi), r * np.cos(phi), np.sqrt(2.0 * (1 - vals[2]))])
  st = np.sin(theta)
  ct = np.cos(theta)
  base_rot = np.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]], dtype=np.float32)
  return (np.outer(v, v) - np.eye(3)).dot(base_rot)


def random_scales(batch_size, minval, maxval):
  scales = tf.random.uniform(
      shape=[batch_size, 3], minval=minval, maxval=maxval)
  hom_coord = tf.ones([batch_size, 1], dtype=tf.float32)
  scales = tf.concat([scales, hom_coord], axis=1)
  s = tf.linalg.diag(scales)
  log.info(s.get_shape().as_list())
  return tf.linalg.diag(scales)


def random_transformation(origin):
  batch_size = origin.get_shape().as_list()[0]
  assert len(origin.get_shape().as_list()) == 2
  center = translation_to_tx(-origin)
  rotate = random_rotations(batch_size)
  scale = random_scales(batch_size, 1, 4)
  tx = tf.matmul(scale, tf.matmul(rotate, center))
  return tx


def random_zoom_transformation(origin):
  batch_size = origin.get_shape().as_list()[0]
  assert len(origin.get_shape().as_list()) == 2
  center = translation_to_tx(-origin)
  scale = random_scales(batch_size, 3, 3)
  tx = tf.matmul(scale, center)
  return tx


def translation_to_tx(t):
  """Maps three translation elements to a 4x4 homogeneous matrix.

  Args:
   t: Tensor with shape [..., 3].

  Returns:
    Tensor with shape [..., 4, 4].
  """
  batch_dims = t.get_shape().as_list()[:-1]
  empty_rot = tf.eye(3, batch_shape=batch_dims)
  rot = tf.concat([empty_rot, t[..., tf.newaxis]], axis=-1)
  hom_row = tf.eye(4, batch_shape=batch_dims)[..., 3:4, :]
  return tf.concat([rot, hom_row], axis=-2)


def rotation_to_tx(rot):
  """Maps a 3x3 rotation matrix to a 4x4 homogeneous matrix.

  Args:
    rot: Tensor with shape [..., 3, 3].

  Returns:
    Tensor with shape [..., 4, 4].
  """
  batch_dims = rot.get_shape().as_list()[:-2]
  empty_col = tf.zeros(batch_dims + [3, 1], dtype=tf.float32)
  rot = tf.concat([rot, empty_col], axis=-1)
  hom_row = tf.eye(4, batch_shape=batch_dims)[..., 3:4, :]
  return tf.concat([rot, hom_row], axis=-2)

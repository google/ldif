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
"""Helpers for differentiable interpolation."""

import numpy as np
import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order


def interpolate_np(grid, samples, world2grid):
  """Returns the trilinearly interpolated SDF values using the grid.

  Args:
    grid: numpy array with shape [depth, height, width].
    samples: numpy array with shape [sample_count, 3].
    world2grid: numpy array with shape [4,4]. Rigid body transform.

  Returns:
    sdf: Tensor with shape [batch_size, sample_count, 1]. The ground truth
      sdf at the sample locations. Differentiable w.r.t. samples.
  """
  xyzw_samples = np.pad(
      samples, [[0, 0], [0, 1]], mode='constant', constant_values=1)
  grid_frame_samples = np.matmul(xyzw_samples, world2grid.T)[..., :3]
  lower_coords = np.floor(grid_frame_samples).astype(np.int32)
  upper_coords = np.ceil(grid_frame_samples).astype(np.int32)
  alpha = grid_frame_samples - lower_coords.astype(np.float32)

  lca = np.split(lower_coords, 3, axis=-1)[::-1]
  uca = np.split(upper_coords, 3, axis=-1)[::-1]
  aca = np.split(alpha, 3, axis=-1)[::-1]  # ?

  c00 = grid[lca[0], lca[1], lca[2]] * (1 - aca[0]) + grid[uca[0], lca[1],
                                                           lca[2]] * aca[0]
  c01 = grid[lca[0], lca[1], uca[2]] * (1 - aca[0]) + grid[uca[0], lca[1],
                                                           uca[2]] * aca[0]
  c10 = grid[lca[0], uca[1], lca[2]] * (1 - aca[0]) + grid[uca[0], uca[1],
                                                           lca[2]] * aca[0]
  c11 = grid[lca[0], uca[1], uca[2]] * (1 - aca[0]) + grid[uca[0], uca[1],
                                                           uca[2]] * aca[0]

  c0 = c00 * (1 - aca[1]) + c10 * aca[1]
  c1 = c01 * (1 - aca[1]) + c11 * aca[1]

  interp = c0 * (1 - aca[2]) + c1 * aca[2]

  log.info('interpolated:')
  log.info(interp.shape)
  log.info(interp)
  log.info('lower coords:')
  log.info(np.min(lower_coords))
  log.info(np.max(lower_coords))
  log.info(np.mean(lower_coords))
  log.info('upper coords:')
  log.info(np.min(upper_coords))
  log.info(np.max(upper_coords))
  log.info(np.mean(upper_coords))
  log.info('Interpolated SDF')
  log.info(np.min(interp))
  log.info(np.max(interp))
  log.info(np.mean(interp))
  log.info('Original SDF')
  log.info(np.min(grid))
  log.info(np.max(grid))
  log.info(np.mean(grid))
  log.info(np.histogram(interp))
  return interp


def ensure_shape(t, s):
  ts = t.get_shape().as_list()
  if len(ts) != len(s):
    raise ValueError('Tensors have rank mismatch: %s vs expected %s' % (ts, s))
  for i, si in enumerate(s):
    if si == -1:
      continue
    if si != ts[i]:
      raise ValueError('Tensors have dimension mismatch: %s expected vs %s' %
                       (ts, s))


def interpolate(grid, samples, world2grid):
  """Returns the trilinearly interpolated function values on the grid.

  Args:
    grid: Tensor with shape [batch_size, depth, height, width]. The function to
      interpolate.
    samples: Tensor with shape [batch_size, sample_count, 3]. The xyz triplets.
    world2grid: Tensor with shape [batch_size, 4, 4]. A rigid body transform
      mapping from sample coordinates to grid coordinates.

  Returns:
    sdf: Tensor with shape [batch_size, sample_count, 1]. The ground truth
      sdf at the sample locations. Differentiable w.r.t. samples.
    invalid: Tensor with shape [batch_size, sample_count, 1] and type tf.bool.
      True where the input samples map outside the supplied grid. The sdf
      tensor will be zero at these locations and there will be no gradient.
  """
  xyzw_samples = tf.pad(
      samples,
      paddings=tf.constant([[0, 0], [0, 0], [0, 1]]),
      mode='CONSTANT',
      constant_values=1)
  ensure_shape(samples, [-1, -1, 3])
  batch_size, sample_count = samples.get_shape().as_list()[:2]
  ensure_shape(grid, [batch_size, -1, -1, -1])
  xyzw_samples = tf.ensure_shape(xyzw_samples, [batch_size, sample_count, 4])
  grid_frame_samples = tf.matmul(
      xyzw_samples, world2grid, transpose_b=True)[..., :3]
  lower_coords = tf.floor(grid_frame_samples)
  alpha = grid_frame_samples - lower_coords
  min_alpha = 1e-05
  max_alpha = 1 - 1e-05
  alpha = tf.clip_by_value(alpha, min_alpha, max_alpha)
  lower_coords = tf.cast(lower_coords, tf.int32)
  upper_coords = tf.cast(tf.ceil(grid_frame_samples), tf.int32)

  depth, height, width = grid.get_shape().as_list()[1:]
  max_vals = np.array([[[width, height, depth]]], dtype=np.int32) - 1
  max_vals = tf.constant(max_vals)
  is_invalid = tf.logical_or(
      tf.reduce_any(lower_coords < 0, axis=-1, keep_dims=True),
      tf.reduce_any(upper_coords > max_vals, axis=-1, keep_dims=True))
  log.info('is_invalid vs lower_coords: %s vs %s' %
           (repr(is_invalid.get_shape().as_list()),
            repr(lower_coords.get_shape().as_list())))
  lower_coords = tf.where_v2(is_invalid, 0, lower_coords)
  log.info('Post-where lower_coords: %s' %
           repr(lower_coords.get_shape().as_list()))
  upper_coords = tf.where_v2(is_invalid, 0, upper_coords)

  lca = tf.split(lower_coords, 3, axis=-1)[::-1]
  uca = tf.split(upper_coords, 3, axis=-1)[::-1]
  aca = tf.unstack(alpha, axis=-1)[::-1]

  lca[0] = tf.ensure_shape(lca[0], [batch_size, sample_count, 1])
  lca[1] = tf.ensure_shape(lca[1], [batch_size, sample_count, 1])
  lca[2] = tf.ensure_shape(lca[2], [batch_size, sample_count, 1])

  batch_indices = np.arange(batch_size, dtype=np.int32)
  batch_indices = np.reshape(batch_indices, [batch_size, 1, 1])
  batch_indices = np.tile(batch_indices, [1, sample_count, 1])
  batch_indices = tf.constant(batch_indices, dtype=tf.int32)

  def batch_gather_nd(source, index_list):
    return tf.gather_nd(source,
                        tf.concat([batch_indices] + index_list, axis=-1))

  def lerp(lval, uval, alpha):
    return lval * (1 - alpha) + uval * (alpha)

  def lookup_and_lerp(lidx, uidx, alpha):
    return lerp(batch_gather_nd(grid, lidx), batch_gather_nd(grid, uidx), alpha)

  c00 = lookup_and_lerp([lca[0], lca[1], lca[2]], [uca[0], lca[1], lca[2]],
                        aca[0])
  c01 = lookup_and_lerp([lca[0], lca[1], uca[2]], [uca[0], lca[1], uca[2]],
                        aca[0])
  c10 = lookup_and_lerp([lca[0], uca[1], lca[2]], [uca[0], uca[1], lca[2]],
                        aca[0])
  c11 = lookup_and_lerp([lca[0], uca[1], uca[2]], [uca[0], uca[1], uca[2]],
                        aca[0])

  c0 = lerp(c00, c10, aca[1])
  c1 = lerp(c01, c11, aca[1])

  sdf = tf.expand_dims(lerp(c0, c1, aca[2]), axis=-1)
  log.info(
      'is_invalid vs sdf coords: %s vs %s' %
      (repr(is_invalid.get_shape().as_list()), repr(sdf.get_shape().as_list())))
  sdf = tf.where_v2(is_invalid, 1e-5, sdf)

  sdf = tf.ensure_shape(sdf, [batch_size, sample_count, 1])
  return sdf, is_invalid


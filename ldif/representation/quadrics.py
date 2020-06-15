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
"""Utilities to evaluate quadric implicit surface functions."""

from ldif.util import camera_util
from ldif.util import tf_util

import tensorflow as tf

NORMALIZATION_EPS = 1e-8
SQRT_NORMALIZATION_EPS = 1e-4
DIV_EPSILON = 1e-8


def sample_quadric_surface(quadric, center, samples):
  """Samples the algebraic distance to the input quadric at sparse locations.


  Args:
    quadric: Tensor with shape [..., 4, 4]. Contains the matrix of the quadric
      surface.
    center: Tensor with shape [..., 3]. Contains the [x,y,z] coordinates of the
      center of the coordinate frame of the quadric surface in NIC space with a
      top-left origin.
    samples: Tensor with shape [..., N, 3], where N is the number of samples to
      evaluate. These are the sample locations in the same space in which the
      quadric surface center is defined. Supports broadcasting the batching
      dimensions.

  Returns:
    distances: Tensor with shape [..., N, 1]. Contains the algebraic distance
      to the surface at each sample.
  """
  with tf.name_scope('sample_quadric_surface'):
    batching_dimensions = quadric.get_shape().as_list()[:-2]
    batching_rank = len(batching_dimensions)
    tf_util.assert_shape(quadric, batching_dimensions + [4, 4],
                         'sample_quadric_surface:quadric')
    tf_util.assert_shape(center, batching_dimensions + [-1],
                         'sample_quadric_surface:center')
    tf_util.assert_shape(samples, batching_rank * [-1] + [-1, 3],
                         'sample_quadric_surface:samples')

    # We want to transform the coordinates so that they are in the coordinate
    # frame of the conic section matrix, so we subtract the center of the
    # conic.
    samples = samples - tf.expand_dims(center, axis=batching_rank)
    sample_count = samples.get_shape().as_list()[-2]

    homogeneous_sample_ones = tf.ones(
        samples.get_shape().as_list()[:-1] + [1], dtype=tf.float32)
    homogeneous_sample_coords = tf.concat([samples, homogeneous_sample_ones],
                                          axis=-1)

    # When we transform the coordinates per-image, we broadcast on both sides-
    # the batching dimensions broadcast up the coordinate grid, and the
    # coordinate center broadcasts up along the height and width.
    # Per-pixel, the algebraic distance is v^T * M * v, where M is the matrix
    # of the conic section, and v is the homogeneous column vector [x y z 1]^T.
    half_distance = tf.matmul(
        quadric, homogeneous_sample_coords, transpose_b=True)
    rank = batching_rank + 2
    half_distance = tf.transpose(
        half_distance, perm=list(range(rank - 2)) + [rank - 1, rank - 2])
    algebraic_distance = tf.reduce_sum(
        tf.multiply(homogeneous_sample_coords, half_distance), axis=-1)
    return tf.reshape(algebraic_distance,
                      batching_dimensions + [sample_count, 1])


def decode_covariance_roll_pitch_yaw(radius, invert=False):
  """Converts 6-D radus vectors to the corresponding covariance matrices.

  Args:
    radius: Tensor with shape [..., 6]. First three numbers are covariances of
      the three Gaussian axes. Second three numbers are the roll-pitch-yaw
      rotation angles of the Gaussian frame.
    invert: Whether to return the inverse covariance.

  Returns:
     Tensor with shape [..., 3, 3]. The 3x3 (optionally inverted) covariance
     matrices corresponding to the input radius vectors.
  """
  d = 1.0 / (radius[..., 0:3] + DIV_EPSILON) if invert else radius[..., 0:3]
  diag = tf.matrix_diag(d)
  rotation = camera_util.roll_pitch_yaw_to_rotation_matrices(radius[..., 3:6])
  return tf.matmul(tf.matmul(rotation, diag), rotation, transpose_b=True)


def sample_cov_bf(center, radius, samples):
  """Samples gaussian radial basis functions at specified coordinates.

  Args:
    center: Tensor with shape [..., 3]. Contains the [x,y,z] coordinates of the
      RBF center in NIC space with a top-left origin.
    radius: Tensor with shape [..., 6]. First three numbers are covariances of
      the three Gaussian axes. Second three numbers are the roll-pitch-yaw
      rotation angles of the Gaussian frame.
    samples: Tensor with shape [..., N, 3],  where N is the number of samples to
      evaluate. These are the sample locations in the same space in which the
      quadric surface center is defined. Supports broadcasting the batching
      dimensions.

  Returns:
     Tensor with shape [..., N, 1]. The basis function strength at each sample
     location.
  """
  with tf.name_scope('sample_cov_bf'):
    # Compute the samples' offset from center, then extract the coordinates.
    diff = samples - tf.expand_dims(center, axis=-2)
    x, y, z = tf.unstack(diff, axis=-1)
    # Decode 6D radius vectors into inverse covariance matrices, then extract
    # unique elements.
    inv_cov = decode_covariance_roll_pitch_yaw(radius, invert=True)
    shape = tf.concat([tf.shape(inv_cov)[:-2], [1, 9]], axis=0)
    inv_cov = tf.reshape(inv_cov, shape)
    c00, c01, c02, _, c11, c12, _, _, c22 = tf.unstack(inv_cov, axis=-1)
    # Compute function value.
    dist = (
        x * (c00 * x + c01 * y + c02 * z) + y * (c01 * x + c11 * y + c12 * z) +
        z * (c02 * x + c12 * y + c22 * z))
    dist = tf.exp(-0.5 * dist)
    return dist


def sample_axis_aligned_bf(center, radius, samples):
  """Samples gaussian radial basis functions at specified coordinates.

  Args:
    center: Tensor with shape [..., 3]. Contains the [x,y,z] coordinates of the
      RBF center in NIC space with a top-left origin.
    radius: Tensor with shape [..., 3]. The covariance of the RBF in NIC space
      along the x, y, and z axes.
    samples: Tensor with shape [..., N, 3],  where N is the number of samples to
      evaluate. These are the sample locations in the same space in which the
      quadric surface center is defined. Supports broadcasting the batching
      dimensions.

  Returns:
     Tensor with shape [..., N, 1]. The basis function strength at each sample
     location.
  """
  with tf.name_scope('sample_axis_aligned_bf'):
    diff = samples - tf.expand_dims(center, axis=-2)
    squared_diff = tf.square(diff)
    scale = tf.minimum((-2) * tf.expand_dims(radius, axis=-2),
                       -NORMALIZATION_EPS)
    return tf.exp(tf.reduce_sum(squared_diff / scale, axis=-1, keepdims=True))


def sample_isotropic_bf(center, radius, samples):
  """Samples gaussian radial basis functions at specified coordinates.

  Args:
    center: Tensor with shape [..., 3]. Contains the [x,y,z] coordinates of the
      RBF center in NIC space with a top-left origin.
    radius: Tensor with shape [..., 1]. Twice the variance of the RBF in NIC
      space.
    samples: Tensor with shape [..., N, 3],  where N is the number of samples to
      evaluate. These are the sample locations in the same space in which the
      quadric surface center is defined. Supports broadcasting the batching
      dimensions.

  Returns:
     Tensor with shape [..., N, 1]. The RBF strength at each sample location.
  """
  with tf.name_scope('sample_isotropic_bf'):
    batching_dimensions = center.get_shape().as_list()[:-1]
    batching_rank = len(batching_dimensions)

    # Reshape the center to allow broadcasting over the sample domain:
    center = tf.expand_dims(center, axis=batching_rank)
    samples -= center
    l2_norm = (
        samples[..., 0] * samples[..., 0] + samples[..., 1] * samples[..., 1] +
        samples[..., 2] * samples[..., 2])
    # Ensure the radius is large enough to avoid numerical issues:
    radius = tf.maximum(SQRT_NORMALIZATION_EPS, radius)
    weights = tf.exp(-0.5 * l2_norm / radius)
    return tf.expand_dims(weights, axis=-1)


def compute_shape_element_influences(quadrics, centers, radii, samples):
  """Computes the per-shape-element values at given sample locations.

  Args:
    quadrics: quadric parameters with shape [batch_size, quadric_count, 4, 4].
    centers: rbf centers with shape [batch_size, quadric_count, 3].
    radii: rbf radii with shape [batch_size, quadric_count, radius_length].
      radius_length can be 1, 3, or 6 depending on whether it is isotropic,
      anisotropic, or a general symmetric covariance matrix, respectively.
    samples: a grid of samples with shape [batch_size, quadric_count,
      sample_count, 3] or shape [batch_size, sample_count, 3].

  Returns:
    Two tensors (the quadric values and the RBF values, respectively), each
    with shape [batch_size, quadric_count, sample_count, 1]
  """
  with tf.name_scope('comptue_shape_element_influences'):
    # Select the number of samples along the ray. The larger this is, the
    # more memory that will be consumed and the slower the algorithm. But it
    # reduces warping artifacts and the likelihood of missing a thin surface.
    batch_size, quadric_count = quadrics.get_shape().as_list()[0:2]

    tf_util.assert_shape(quadrics, [batch_size, quadric_count, 4, 4],
                         'quadrics')
    tf_util.assert_shape(centers, [batch_size, quadric_count, 3], 'centers')
    # We separate the isometric, axis-aligned, and general RBF functions.
    # The primary reason for this is that the type of basis function
    # affects the shape of many tensors, and it is easier to make
    # everything correct when the shape is known. Once the shape function is
    # set we can clean it up and choose one basis function.
    radii_shape = radii.get_shape().as_list()
    if len(radii_shape) != 3:
      raise tf.errors.InvalidArgumentError(
          'radii must have shape [batch_size, quadric_count, radii_values].')
    elif radii_shape[2] == 1:
      rbf_sampler = sample_isotropic_bf
      radius_shape = [1]
    elif radii_shape[2] == 3:
      rbf_sampler = sample_axis_aligned_bf
      radius_shape = [3]
    elif radii_shape[2] == 6:
      rbf_sampler = sample_cov_bf
      radius_shape = [6]
    else:
      raise tf.errors.InvalidArgumentError(
          'radii must have either 1, 3, or 6 elements.')
    tf_util.assert_shape(radii, [batch_size, quadric_count] + radius_shape,
                         'radii')

    # Ensure the samples have the right shape and tile in an axis for the
    # quadric dimension if it wasn't provided.
    sample_shape = samples.get_shape().as_list()
    sample_rank = len(sample_shape)
    if (sample_rank not in [3, 4] or sample_shape[-1] != 3 or
        sample_shape[0] != batch_size):
      raise tf.errors.InvalidArgumentError(
          'Input tensor samples must have shape [batch_size, quadric_count,'
          ' sample_count, 3] or shape [batch_size, sample_count, 3]. The input'
          ' shape was %s' % repr(sample_shape))
    missing_quadric_dim = len(sample_shape) == 3
    if missing_quadric_dim:
      samples = tf_util.tile_new_axis(samples, axis=1, length=quadric_count)
    sample_count = sample_shape[-2]

    # Sample the quadric surfaces and the RBFs in world space, and composite
    # them.
    sampled_quadrics = sample_quadric_surface(quadrics, centers, samples)
    tf_util.assert_shape(sampled_quadrics,
                         [batch_size, quadric_count, sample_count, 1],
                         'sampled_quadrics')

    tf_util.assert_shape(centers, [batch_size, quadric_count, 3], 'centers')
    tf_util.assert_shape(samples, [batch_size, quadric_count, sample_count, 3],
                         'samples')
    sampled_rbfs = rbf_sampler(centers, radii, samples)
    sampled_rbfs = tf.reshape(sampled_rbfs,
                              [batch_size, quadric_count, sample_count, 1])
    return sampled_quadrics, sampled_rbfs

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
"""Tensorflow functions for replicating GAPS operations."""

import importlib
import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import geom_util
from ldif.util import np_util
from ldif.util import tf_util
# pylint: enable=g-bad-import-order

importlib.reload(tf_util)


def gaps_depth_image_to_cam_image(depth_image, xfov):
  """Converts a GAPS depth image tensor to a camera-space image tensor.

  Args:
    depth_image: Tensor with shape [batch_size, height, width, 1].
    xfov: Scalar or tensor with shape [1] or [batch_size].

  Returns:
    cam_image: Tensor with shape [batch_size, height, width, 3].
  """
  batch_size, height, width = depth_image.get_shape().as_list()[:3]
  depth_image = tf.ensure_shape(depth_image, [batch_size, height, width, 1])
  if isinstance(xfov, float):
    xfov = tf.constant([xfov], dtype=tf.float32)
    xfov = tf.tile(xfov, [batch_size])
  else:
    xfov = tf.reshape(xfov, [batch_size])
#  if xfov.get_shape().as_list()[0] == 1:
#    xfov = tf.tile(xfov, [batch_size])
#  else:
#    assert xfov.get_shape().as_list()[0] == batch_size

  pixel_coords = np_util.make_coordinate_grid(
      height, width, is_screen_space=False, is_homogeneous=False)
  # Values should go from -1 -> 1, not from 0 -> 1:
  nic_x = np_util.batch_np(2 * pixel_coords[:, :, 0:1] - 1.0, batch_size)
  nic_y = np_util.batch_np(2 * pixel_coords[:, :, 1:2] - 1.0, batch_size)
  nic_d = -depth_image
  aspect = height / float(width)
  tan_xfov = tf.math.tan(xfov)
  yfov = tf.math.atan(aspect * tan_xfov)
  intrinsics_00 = tf.reshape(1.0 / tan_xfov, [batch_size, 1, 1, 1])
  intrinsics_11 = tf.reshape(1.0 / tf.math.tan(yfov), [batch_size, 1, 1, 1])
  cam_x = nic_x * -nic_d / intrinsics_00
  cam_y = nic_y * nic_d / intrinsics_11
  cam_z = nic_d
  return tf.concat([cam_x, cam_y, cam_z], axis=3)


def gaps_depth_image_to_xyz_image(depth_image, xfov, cam2world, mask=None):
  """Converts a GAPS depth image to world space.

  Args:
    depth_image: Tensor with shape [batch_size, height, width, 1].
    xfov: Scalar or Tensor with shape [1] or [batch_size].
    cam2world: Transformation matrix Tensor with shape [batch_size, 4, 4].
    mask: If provided, a Tensor with shape [batch_size, height, width, 1] that
      is of type bool and is true where the image is considered valid.

  Returns:
    Tensor with shape [batch_size, height, width, 3].
  """
  cam_images = gaps_depth_image_to_cam_image(depth_image, xfov)
  xyz_images = geom_util.apply_4x4(cam_images, cam2world, are_points=True,
                                   batch_rank=1, sample_rank=2)
  if mask is not None:
    xyz_images = tf_util.zero_by_mask(mask, xyz_images, replace_with=0.0)
  return xyz_images

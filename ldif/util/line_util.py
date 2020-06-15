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
"""Functions for differentiable line drawing."""

import numpy as np
import tensorflow as tf


def line_to_image(line_parameters, height, width, falloff=2.0):
  """Renders a 'line' (rectangle) image from its parameterization.

  Args:
    line_parameters: Tensor with shape [5]. Contains the angle of rotation, x
      center, y center, x thickness, and y thickness in order. Coordinates are
      specified in radians and screen space, respectively, with a top left
      origin.
    height: Int containing height in pixels of the desired output render.
    width: Int containing width in pixels of the desired output render.
    falloff: Float containing the soft falloff parameter for the line. Bigger
      values indicate a longer fade into grey outside the line borders. If None
      is passed instead, the line will be drawn with sharp lines.

  Returns:
    Tensor with shape [height, width, 1] containing the line image. Colors are
    in the range [0, 1]- 0 is entirely inside the line, and 0 is entirely
    outside the line.
  """
  with tf.name_scope('line-to-image'):
    # Initialize constant coordinates to be hit-tested:
    x_coords = np.linspace(0.5, width - 0.5, width)
    y_coords = np.linspace(0.5, height - 0.5, height)
    grid_x, grid_y = np.meshgrid(
        x_coords, y_coords, sparse=False, indexing='xy')
    coords = np.stack([grid_x, grid_y], axis=2)
    coords = tf.constant(coords, dtype=tf.float32)

    # Construct rectangle from input parameters:
    angle_of_rotation, px, py, lx, ly = tf.unstack(line_parameters)
    angle_of_rotation = -angle_of_rotation
    center = line_parameters[1:3]

    v0 = tf.stack([px + lx, py + ly], axis=0)
    v0 = rotate_about_point(angle_of_rotation, center, v0)

    v1 = tf.stack([px + lx, py - ly], axis=0)
    v1 = rotate_about_point(angle_of_rotation, center, v1)

    v2 = tf.stack([px - lx, py - ly], axis=0)
    v2 = rotate_about_point(angle_of_rotation, center, v2)

    coords = tf.reshape(coords, [height * width, 1, 2])

    first_direction_insidedness = fractional_vector_projection(
        v1, v0, coords, falloff=falloff)
    second_direction_insidedness = fractional_vector_projection(
        v1, v2, coords, falloff=falloff)
    crease_corners = True
    if crease_corners:
      insidedness = first_direction_insidedness * second_direction_insidedness
    else:
      # Default to a euclidean distance; this is only valid if the insidedness
      # functions are also euclidean.
      insidedness = tf.maximum(
          1.0 - tf.sqrt((1.0 - first_direction_insidedness) *
                        (1.0 - first_direction_insidedness) +
                        (1.0 - second_direction_insidedness) *
                        (1.0 - second_direction_insidedness)),
          tf.zeros_like(first_direction_insidedness))

    color = 1.0 - insidedness
    return tf.reshape(color, [height, width, 1])


def fractional_vector_projection(e0, e1, p, falloff=2.0):
  """Returns a fraction describing whether p projects inside the segment e0 e1.

  If p projects inside the segment, the result is 1. If it projects outside,
  the result is a fraction that is always greater than 0 but monotonically
  decreasing as the distance to the inside of the segment increase.

  Args:
    e0: Tensor with two elements containing the first endpoint XY locations.
    e1: Tensor with two elements containing the second endpoint XY locations.
    p: Tensor with shape [batch_size, 2] containing the query points.
    falloff: Float or Scalar Tensor specifying the softness of the falloff of
      the projection. Larger means a longer falloff.
  """
  with tf.name_scope('fractional-vector-projection'):
    batch_size = p.shape[0].value
    p = tf.reshape(p, [batch_size, 2])
    e0 = tf.reshape(e0, [1, 2])
    e1 = tf.reshape(e1, [1, 2])
    e01 = e1 - e0
    # Normalize for vector projection:
    e01_norm = tf.sqrt(e01[0, 0] * e01[0, 0] + e01[0, 1] * e01[0, 1])
    e01_normalized = e01 / tf.reshape(e01_norm, [1, 1])
    e0p = p - e0
    e0p_dot_e01_normalized = tf.matmul(
        tf.reshape(e0p, [1, batch_size, 2]),
        tf.reshape(e01_normalized, [1, 1, 2]),
        transpose_b=True)
    e0p_dot_e01_normalized = tf.reshape(e0p_dot_e01_normalized,
                                        [batch_size]) / e01_norm
    if falloff is None:
      left_sided_inside = tf.cast(
          tf.logical_and(e0p_dot_e01_normalized >= 0,
                         e0p_dot_e01_normalized <= 1),
          dtype=tf.float32)
      return left_sided_inside

    # Now that we have done the left side, do the right side:
    e10_normalized = -e01_normalized
    e1p = p - e1
    e1p_dot_e10_normalized = tf.matmul(
        tf.reshape(e1p, [1, batch_size, 2]),
        tf.reshape(e10_normalized, [1, 1, 2]),
        transpose_b=True)
    e1p_dot_e10_normalized = tf.reshape(e1p_dot_e10_normalized,
                                        [batch_size]) / e01_norm

    # Take the maximum of the two projections so we face it from the positive
    # direction:
    proj = tf.maximum(e0p_dot_e01_normalized, e1p_dot_e10_normalized)
    proj = tf.maximum(proj, 1.0)

    # A projection value of 1 means at the border exactly.
    # Take the max with 1, to throw out all cases besides 'left' overhang.
    falloff_is_relative = True
    if falloff_is_relative:
      fractional_falloff = 1.0 / (tf.pow(falloff * (proj - 1), 2.0) + 1.0)
      return fractional_falloff
    else:
      # Currently the proj value is given as a distance that is the fraction of
      # the length of the line. Instead, multiply by the length of the line
      # to get the distance in pixels. Then, set a target '0' distance, (i.e.
      # 10 pixels). Divide by that distance so we express distance in multiples
      # of the max distance that gets seen.
      # threshold at 1, and return 1 - that to get linear falloff from 0 to
      # the target distance.
      line_length = tf.reshape(e01_norm, [1])
      pixel_dist = tf.reshape(proj - 1, [-1]) * line_length
      zero_thresh_in_pixels = tf.reshape(
          tf.constant([8.0], dtype=tf.float32), [1])
      relative_dist = pixel_dist / zero_thresh_in_pixels
      return 1.0 / (tf.pow(relative_dist, 3.0) + 1.0)


def rotate_about_point(angle_of_rotation, point, to_rotate):
  """Rotates a single input 2d point by a specified angle around a point."""
  with tf.name_scope('rotate-2d'):
    cos_angle = tf.cos(angle_of_rotation)
    sin_angle = tf.sin(angle_of_rotation)
    top_row = tf.stack([cos_angle, -sin_angle], axis=0)
    bottom_row = tf.stack([sin_angle, cos_angle], axis=0)
    rotation_matrix = tf.reshape(
        tf.stack([top_row, bottom_row], axis=0), [1, 2, 2])
    to_rotate = tf.reshape(to_rotate, [1, 1, 2])
    point = tf.reshape(point, [1, 1, 2])
    to_rotate = to_rotate - point
    to_rotate = tf.matmul(rotation_matrix, to_rotate, transpose_b=True)
    to_rotate = tf.reshape(to_rotate, [1, 1, 2]) + point
    return to_rotate


def union_of_line_drawings(lines):
  """Computes the union image of a sequence of line predictions."""
  with tf.name_scope('Union-of-Line-Images'):
    lines = tf.stack(lines, axis=-1)
    lines = tf.reduce_min(lines, axis=-1)
    return lines


def network_line_parameters_to_line(line_parameters, height, width):
  """Interprets a network's output as line parameters and calls line_to_image.

  Rescales to assume the network output is not resolution dependent, and
  clips to valid parameters.

  Args:
    line_parameters: Tensor with shape [batch_size, 5]. Contains the network
      output, to be interpreted as line parameters.
    height: Int containing output height.
    width: Int containing output width.

  Returns:
    An image with shape [batch_size, height, width, 1]. Contains a drawing of
    the network's output.
  """
  rotation, px, py, lx, ly = tf.unstack(line_parameters, axis=1)
  px = tf.minimum(tf.nn.relu(px * width + width / 2), width)  # was leaky relu!
  py = tf.minimum(tf.nn.relu(py * height + height / 2),
                  height)  # was leaky relu!
  lx = tf.clip_by_value(tf.abs(lx) * width, 4.0, width / 3.0)
  ly = tf.clip_by_value(tf.abs(ly) * height, 4.0, height / 3.0)
  line_parameters = tf.stack([rotation, px, py, lx, ly], axis=1)
  batch_out = []
  for batch_item in tf.unstack(line_parameters, axis=0):
    batch_out.append(line_to_image(batch_item, height, width))
  return tf.stack(batch_out)

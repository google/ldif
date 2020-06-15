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
"""Utilities for geometric operations in tensorflow."""

import math

import numpy as np
import tensorflow as tf

# ldif is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import np_util
from ldif.util import camera_util
from ldif.util import tf_util
# pylint: enable=g-bad-import-order


def ray_sphere_intersect(ray_start, ray_direction, sphere_center, sphere_radius,
                         max_t):
  """Intersect rays with each of a set of spheres.

  Args:
    ray_start: Tensor with shape [batch_size, ray_count, 3]. The end point of
      the rays. In the same coordinate space as the spheres.
    ray_direction: Tensor with shape [batch_size, ray_count, 3]. The extant ray
      direction.
    sphere_center: Tensor with shape [batch_size, sphere_count, 3]. The center
      of the spheres.
    sphere_radius: Tensor with shape [batch_size, sphere_count, 1]. The radius
      of the spheres.
    max_t: The maximum intersection distance.

  Returns:
    intersections: Tensor with shape [batch_size, ray_count, sphere_count]. If
      no intersection is found between [0, max_t), then the value will be max_t.
  """
  # We apply the algebraic solution:
  batch_size, ray_count = ray_start.get_shape().as_list()[:2]
  sphere_count = sphere_center.get_shape().as_list()[1]
  ray_direction = tf.reshape(ray_direction, [batch_size, ray_count, 1, 3])
  ray_start = tf.reshape(ray_start, [batch_size, ray_count, 1, 3])
  sphere_center = tf.reshape(sphere_center, [batch_size, 1, sphere_count, 3])
  sphere_radius = tf.reshape(sphere_radius, [batch_size, 1, sphere_count, 1])
  a = 1.0
  b = 2.0 * ray_direction * (ray_start - sphere_center)
  ray_sphere_distance = tf.reduce_sum(
      tf.square(ray_start - sphere_center), axis=-1, keep_dims=True)
  c = ray_sphere_distance - tf.square(sphere_radius)
  discriminant = tf.square(b) - 4 * a * c
  # Assume it's positive, then zero out later:
  ta = tf.divide((-b + tf.sqrt(discriminant)), 2 * a)
  tb = tf.divide((-b - tf.sqrt(discriminant)), 2 * a)
  t0 = tf.minimum(ta, tb)
  t1 = tf.maximum(ta, tb)
  t = tf.where(t0 > 0, t0, t1)
  intersection_invalid = tf.logical_or(
      tf.logical_or(discriminant < 0, t < 0), t > max_t)
  t = tf.where(intersection_invalid, max_t * tf.ones_like(t), t)
  return t


def to_homogeneous(t, is_point):
  """Makes a homogeneous space tensor given a tensor with ultimate coordinates.

  Args:
    t: Tensor with shape [..., K], where t is a tensor of points in
      K-dimensional space.
    is_point: Boolean. True for points, false for directions

  Returns:
    Tensor with shape [..., K+1]. t padded to be homogeneous.
  """
  padding = 1 if is_point else 0
  rank = len(t.get_shape().as_list())
  paddings = []
  for _ in range(rank):
    paddings.append([0, 0])
  paddings[-1][1] = 1
  return tf.pad(
      t, tf.constant(paddings), mode='CONSTANT', constant_values=padding)


def transform_points_with_normals(points, tx, normals=None):
  """Transforms a pointcloud with normals to a new coordinate frame.

  Args:
    points: Tensor with shape [batch_size, point_count, 3 or 6].
    tx: Tensor with shape [batch_size, 4, 4]. Takes column-vectors from the
      current frame to the new frame as T*x.
    normals: Tensor with shape [batch_size, point_count, 3] if provided. None
      otherwise. If the points tensor contains normals, this should be None.

  Returns:
    If tensor 'points' has shape [..., 6], then a single tensor with shape
      [..., 6] in the new frame. If 'points' has shape [..., 3], then returns
      either one or two tensors of shape [..., 3] depending on whether 'normals'
      is None.
  """
  if len(points.shape) != 3:
    raise ValueError(f'Invalid points shape: {points.get_shape().as_list()}')
  if len(tx.shape) != 3:
    raise ValueError(f'Invalid tx shape: {tx.get_shape().as_list()}')
  are_concatenated = points.shape[-1] == 6
  if are_concatenated:
    points, normals = tf.split(points, [3, 3], axis=-1)

  transformed_samples = apply_4x4(
      points, tx, are_points=True, batch_rank=1, sample_rank=1)
  if normals is not None:
    transformed_normals = apply_4x4(
        normals,
        tf.linalg.inv(tf.transpose(tx, perm=[0, 2, 1])),
        are_points=False,
        batch_rank=1,
        sample_rank=1)
    transformed_normals = transformed_normals / (
        tf.linalg.norm(transformed_normals, axis=-1, keepdims=True) + 1e-8)
  if are_concatenated:
    return tf.concat([transformed_samples, transformed_normals], axis=-1)
  if normals is not None:
    return transformed_samples, transformed_normals
  return transformed_samples


def transform_featured_points(points, tx):
  """Transforms a pointcloud with features.

  Args:
    points: Tensor with shape [batch_size, point_count, 3+feature_count].
    tx: Tensor with shape [batch_size, 4, 4].

  Returns:
    Tensor with shape [batch_size, point_count, 3+feature_count].
  """
  feature_count = points.get_shape().as_list()[-1] - 3
  if feature_count == 0:
    xyz = points
    features = None
  else:
    xyz, features = tf.split(points, [3, feature_count], axis=2)

  xyz = apply_4x4(xyz, tx, are_points=True, batch_rank=1, sample_rank=1)
  if feature_count:
    return tf.concat([xyz, features], axis=2)
  return xyz


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


def extract_points_near_origin(points, count, features=None):
  """Returns the points nearest to the origin in a pointcloud.

  Args:
    points: Tensor with shape [batch_size, point_count, 3 or more].
    count: The number of points to extract.
    features: Tensor with shape [batch_size, point_count, feature_count] if
      present. None otherwise.

  Returns:
    Either one tensor of size [batch_size, count, 3 or 6] or two tensors of
    size [batch_size, count, 3], depending on whether normals was provided and
    the shape of the 'points' tensor.
  """
  are_concatenated = points.get_shape().as_list()[-1] > 3
  if are_concatenated:
    feature_count = points.get_shape().as_list()[-1] - 3
    original = points
    points, features = tf.split(points, [3, feature_count], axis=-1)
  else:
    assert points.get_shape().as_list()[-1] == 3

  candidate_dists = tf.linalg.norm(points, axis=-1)
  _, selected_indices = tf.math.top_k(-candidate_dists, k=count, sorted=False)
  if are_concatenated:
    return tf.gather(original, selected_indices, batch_dims=1)
  else:
    selected_points = tf.gather(points, selected_indices, batch_dims=1)
    if features is not None:
      return selected_points, tf.gather(
          features, selected_indices, batch_dims=1)
    return selected_points


def local_views_of_shape(global_points,
                         world2local,
                         local_point_count,
                         global_normals=None,
                         global_features=None,
                         zeros_invalid=False,
                         zero_threshold=1e-6,
                         expand_region=True,
                         threshold=4.0):
  """Computes a set of local point cloud observations from a global observation.

  It is assumed for optimization purposes that
  global_point_count >> local_point_count.

  Args:
    global_points: Tensor with shape [batch_size, global_point_count, 3]. The
      input observation point cloud in world space.
    world2local: Tensor with shape [batch_size, frame_count, 4, 4]. Each 4x4
      matrix maps from points in world space to points in a local frame.
    local_point_count: Integer. The number of points to output in each local
      frame. Whatever this value, the local_point_count closest points to each
      local frame origin will be returned.
    global_normals: Tensor with shape [batch_size, global_point_count, 3]. The
      input observation point cloud's normals in world space. Optional.
    global_features: Tensor with shape [batch_size, global_point_count,
      feature_count]. The input observation point cloud features, in any space.
      Optional.
    zeros_invalid: Whether to consider the vector [0, 0, 0] to be invalid.
    zero_threshold: Values less than this in magnitude are considered to be 0.
    expand_region: Whether to expand outward from the threshold region. If
      false, fill with zeros.
    threshold: The distance threshold.

  Returns:
    local_points: Tensor with shape [batch_size, frame_count,
      local_point_count, 3].
    local_normals: Tensor with shape [batch_size, frame_count,
      local_point_count, 3]. None if global_normals not provided.
    local_features: Tensor with shape [batch_size, frame_count,
      local_point_count, feature_count]. Unlike the local normals and points,
      these are not transformed because there may or may not be a good
      transformation to apply, depending on what the features are. But they will
      be the features associated with the local points that were chosen. None
      if global_features not provided.
  """
  # Example use case: batch_size = 64, global_point_count = 100000
  # local_point_count = 1000, frame_count = 25. Then:
  # global_points has size 64*100000*3*4 = 73mb
  # local_points has size 64*1000*25*3*4 = 18mb
  # If we made an intermediate tensor with shape [batch_size, frame_count,
  #   global_point_count, 3] -> 64 * 25 * 100000 * 3 * 4 = 1.8 Gb -> bad.

  batch_size, _, _ = global_points.get_shape().as_list()
  if zeros_invalid:
    # If we just set the global points to be very far away, they won't be a
    # nearest neighbor
    abs_zero = False
    if abs_zero:
      is_zero = tf.reduce_all(
          tf.equal(global_points, 0.0), axis=-1, keepdims=True)
    else:
      is_zero = tf.reduce_all(
          tf.abs(global_points) < zero_threshold, axis=-1, keepdims=True)
    global_points = tf.where_v2(is_zero, 100.0, global_points)
  _, frame_count, _, _ = world2local.get_shape().as_list()

  local2world = tf.matrix_inverse(world2local)

  # *sigh* oh well, guess we have to do the transform:
  tiled_global = tf.tile(
      tf.expand_dims(to_homogeneous(global_points, is_point=True), axis=1),
      [1, frame_count, 1, 1])
  all_local_points = tf.matmul(tiled_global, world2local, transpose_b=True)
  distances = tf.norm(all_local_points, axis=-1)
  # thresh = 4.0
  # TODO(kgenova) This is potentially a problem because it could introduce
  # randomness into the pipeline at inference time.
  probabilities = tf.random.uniform(distances.get_shape().as_list())
  is_valid = distances < threshold

  sample_order = tf.where(is_valid, probabilities, -distances)
  _, top_indices = tf.math.top_k(
      sample_order, k=local_point_count, sorted=False)
  local_points = tf.gather(all_local_points, top_indices, batch_dims=2, axis=-2)
  local_points = tf.ensure_shape(
      local_points[..., :3], [batch_size, frame_count, local_point_count, 3])
  is_valid = tf.expand_dims(is_valid, axis=-1)
  # log.info('is_valid shape: ', is_valid.get_shape().as_list())
  # log.info('top_indices shape: ', top_indices.get_shape().as_list())
  # log.info('all_local_points shape: ', all_local_points.get_shape().as_list())
  points_valid = tf.gather(is_valid, top_indices, batch_dims=2, axis=-2)
  # points_valid = tf.expand_dims(points_valid, axis=-1)
  points_valid = tf.ensure_shape(
      points_valid, [batch_size, frame_count, local_point_count, 1])
  if not expand_region:
    local_points = tf.where_v2(points_valid, local_points, 0.0)
  # valid_feature = tf.cast(points_valid, dtype=tf.float32)

  if global_normals is not None:
    tiled_global_normals = tf.tile(
        tf.expand_dims(to_homogeneous(global_normals, is_point=False), axis=1),
        [1, frame_count, 1, 1])
    # Normals get transformed by the inverse-transpose matrix:
    all_local_normals = tf.matmul(
        tiled_global_normals, local2world, transpose_b=False)
    local_normals = tf.gather(
        all_local_normals, top_indices, batch_dims=2, axis=-2)
    # Remove the homogeneous coordinate now. It isn't a bug to normalize with
    # it since it's zero, but it's confusing.
    local_normals = tf.math.l2_normalize(local_normals[..., :3], axis=-1)
    local_normals = tf.ensure_shape(
        local_normals, [batch_size, frame_count, local_point_count, 3])
  else:
    local_normals = None

  if global_features is not None:
    feature_count = global_features.get_shape().as_list()[-1]
    local_features = tf.gather(
        global_features, top_indices, batch_dims=1, axis=-2)
    local_features = tf.ensure_shape(
        local_features,
        [batch_size, frame_count, local_point_count, feature_count])
  else:
    local_features = None
  return local_points, local_normals, local_features, points_valid


def chamfer_distance(pred, target):
  """Computes the chamfer distance between two point sets, in both directions.

  Args:
    pred: Tensor with shape [..., pred_point_count, n_dims].
    target: Tensor with shape [..., target_point_count, n_dims].

  Returns:
    pred_to_target, target_to_pred.
    pred_to_target: Tensor with shape [..., pred_point_count, 1]. The distance
      from each point in pred to the closest point in the target.
    target_to_pred: Tensor with shape [..., target_point_count, 1]. The distance
      from each point in target to the closet point in the prediction.
  """
  with tf.name_scope('chamfer_distance'):
    # batching_dimensions = pred.get_shape().as_list()[:-2]
    # batching_rank = len(batching_dimensions)
    # pred_norm_squared = tf.matmul(pred, pred, transpose_b=True)
    # target_norm_squared = tf.matmul(target, target, transpose_b=True)
    # target_mul_pred_t = tf.matmul(pred, target, transpose_b=True)
    # pred_mul_target_t = tf.matmul(target, pred, transpose_b=True)

    differences = tf.expand_dims(
        pred, axis=-2) - tf.expand_dims(
            target, axis=-3)
    squared_distances = tf.reduce_sum(differences * differences, axis=-1)
    # squared_distances = tf.matmul(differences, differences, transpose_b=True)
    # differences = pred - tf.transpose(target, perm=range(batching_rank) +
    #  [batching_rank+2, batching_rank+1])
    pred_to_target = tf.reduce_min(squared_distances, axis=-1)
    target_to_pred = tf.reduce_min(squared_distances, axis=-2)
    pred_to_target = tf.expand_dims(pred_to_target, axis=-1)
    target_to_pred = tf.expand_dims(target_to_pred, axis=-1)
    return tf.sqrt(pred_to_target), tf.sqrt(target_to_pred)


def dodeca_parameters(dodeca_idx):
  """Computes the viewpoint, centroid, and up vectors for the dodecahedron."""
  gr = (1.0 + math.sqrt(5.0)) / 2.0
  rgr = 1.0 / gr
  viewpoints = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1],
                [-1, 1, -1], [-1, -1, 1], [-1, -1, -1], [0, gr, rgr],
                [0, gr, -rgr], [0, -gr, rgr], [0, -gr, -rgr], [rgr, 0, gr],
                [rgr, 0, -gr], [-rgr, 0, gr], [-rgr, 0, -gr], [gr, rgr, 0],
                [gr, -rgr, 0], [-gr, rgr, 0], [-gr, -rgr, 0]]
  viewpoint = 0.6 * np.array(viewpoints[dodeca_idx], dtype=np.float32)
  centroid = np.array([0., 0., 0.], dtype=np.float32)
  world_up = np.array([0., 1., 0.], dtype=np.float32)
  return viewpoint, centroid, world_up


def get_camera_to_world(viewpoint, center, world_up):
  """Computes a 4x4 mapping from camera space to world space."""
  towards = center - viewpoint
  towards = towards / np.linalg.norm(towards)
  right = np.cross(towards, world_up)
  right = right / np.linalg.norm(right)
  cam_up = np.cross(right, towards)
  cam_up = cam_up / np.linalg.norm(cam_up)
  rotation = np.stack([right, cam_up, -towards], axis=1)
  rotation_4x4 = np.eye(4)
  rotation_4x4[:3, :3] = rotation
  camera_to_world = rotation_4x4.copy()
  camera_to_world[:3, 3] = viewpoint
  return camera_to_world


def get_dodeca_camera_to_worlds():
  camera_to_worlds = []
  for i in range(20):
    camera_to_worlds.append(get_camera_to_world(*dodeca_parameters(i)))
  camera_to_worlds = np.stack(camera_to_worlds, axis=0)
  return camera_to_worlds


def gaps_depth_render_to_xyz(model_config, depth_image, camera_parameters):
  """Transforms a depth image to camera space assuming its dodeca parameters."""
  # TODO(kgenova) Extract viewpoint, width, height from camera parameters.
  del camera_parameters
  depth_image_height, depth_image_width = depth_image.get_shape().as_list()[1:3]
  if model_config.hparams.didx == 0:
    viewpoint = np.array([1.03276, 0.757946, -0.564739])
    towards = np.array([-0.737684, -0.54139, 0.403385])  #  = v/-1.4
    up = np.array([-0.47501, 0.840771, 0.259748])
  else:
    assert False
  towards = towards / np.linalg.norm(towards)
  right = np.cross(towards, up)
  right = right / np.linalg.norm(right)
  up = np.cross(right, towards)
  up = up / np.linalg.norm(up)
  rotation = np.stack([right, up, -towards], axis=1)
  rotation_4x4 = np.eye(4)
  rotation_4x4[:3, :3] = rotation
  camera_to_world = rotation_4x4.copy()
  camera_to_world[:3, 3] = viewpoint
  camera_to_world = tf.constant(camera_to_world.astype(np.float32))
  world_to_camera = tf.reshape(tf.matrix_inverse(camera_to_world), [1, 4, 4])
  world_to_camera = tf.tile(world_to_camera, [model_config.hparams.bs, 1, 1])
  xyz_image, _, _ = depth_image_to_xyz_image(
      depth_image, world_to_camera, xfov=0.5)
  xyz_image = tf.reshape(
      xyz_image,
      [model_config.hparams.bs, depth_image_height, depth_image_width, 3])
  return xyz_image


def angle_of_rotation_to_2d_rotation_matrix(angle_of_rotation):
  """Given a batch of rotations, create a batch of 2d rotation matrices.

  Args:
    angle_of_rotation: Tensor with shape [batch_size].

  Returns:
    Tensor with shape [batch_size, 2, 2]
  """
  c = tf.cos(angle_of_rotation)
  s = tf.sin(angle_of_rotation)
  top_row = tf.stack([c, -s], axis=1)
  bottom_row = tf.stack([s, c], axis=1)
  return tf.stack([top_row, bottom_row], axis=1)


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


def interpolate_from_grid(samples, grid):
  grid_coordinates = (samples + 0.5) * 63.0
  return interpolate_from_grid_coordinates(grid_coordinates, grid)


def reflect(samples, reflect_x=False, reflect_y=False, reflect_z=False):
  """Reflects the sample locations across the planes specified in xyz.

  Args:
    samples: Tensor with shape [..., 3].
    reflect_x: Bool.
    reflect_y: Bool.
    reflect_z: Bool.

  Returns:
    Tensor with shape [..., 3]. The reflected samples.
  """
  assert isinstance(reflect_x, bool)
  assert isinstance(reflect_y, bool)
  assert isinstance(reflect_z, bool)
  floats = [-1.0 if ax else 1.0 for ax in [reflect_x, reflect_y, reflect_z]]
  mult = np.array(floats, dtype=np.float32)
  shape = samples.get_shape().as_list()
  leading_dims = shape[:-1]
  assert shape[-1] == 3
  mult = mult.reshape([1] * len(leading_dims) + [3])
  mult = tf.constant(mult, dtype=tf.float32)
  return mult * samples


def z_reflect(samples):
  """Reflects the sample locations across the XY plane.

  Args:
    samples: Tensor with shape [..., 3]

  Returns:
    reflected: Tensor with shape [..., 3]. The reflected samples.
  """
  return reflect(samples, reflect_z=True)


def get_world_to_camera(idx):
  assert idx == 1
  eye = tf.constant([[0.671273, 0.757946, -0.966907]], dtype=tf.float32)
  look_at = tf.zeros_like(eye)
  world_up = tf.constant([[0., 1., 0.]], dtype=tf.float32)
  world_to_camera = camera_util.look_at(eye, look_at, world_up)
  return world_to_camera


def transform_depth_dodeca_to_xyz_dodeca(depth_dodeca):
  """Lifts a dodecahedron of depth images to world space."""
  batch_size = depth_dodeca.get_shape().as_list()[0]
  cam2world = get_dodeca_camera_to_worlds()
  cam2world = np.reshape(cam2world, [1, 20, 4, 4]).astype(np.float32)
  world2cams = np.linalg.inv(cam2world)
  world2cams = np.tile(world2cams, [batch_size, 1, 1, 1])
  world2cams = tf.unstack(tf.constant(world2cams, dtype=tf.float32), axis=1)
  depth_im_stack = tf.unstack(depth_dodeca, axis=1)
  assert len(depth_im_stack) == 20
  assert len(world2cams) == 20
  xyz_images = []
  for i in range(20):
    world2cam = world2cams[i]
    depth_im = depth_im_stack[i]
    xyz_image = depth_image_to_xyz_image(depth_im, world2cam, xfov=0.5)[0]
    xyz_images.append(xyz_image)
  xyz_images = tf.stack(xyz_images, axis=1)
  xyz_images = tf.where_v2(depth_dodeca > 0.0, xyz_images, 0.0)
  return xyz_images


def transform_depth_dodeca_to_xyz_dodeca_np(depth_dodeca):
  graph = tf.Graph()
  with graph.as_default():
    depth_in = tf.constant(depth_dodeca)
    xyz_out = transform_depth_dodeca_to_xyz_dodeca(depth_in)
    with tf.Session() as session:
      out_np = session.run(xyz_out)
  return out_np


def _unbatch(arr):
  if arr.shape[0] == 1:
    return arr.reshape(arr.shape[1:])
  return arr


def to_homogenous_np(arr, is_point=True):
  assert arr.shape[-1] in [2, 3]
  homogeneous_shape = list(arr.shape[:-1]) + [1]
  if is_point:
    coord = np.ones(homogeneous_shape, dtype=np.float32)
  else:
    coord = np.zeros(homogeneous_shape, dtype=np.float32)
  return np.concatenate([arr, coord], axis=-1)


def depth_to_cam_np(im, xfov=0.5):
  """Converts a gaps depth image to camera space."""
  im = _unbatch(im)
  height, width, _ = im.shape
  pixel_coords = np_util.make_coordinate_grid(
      height, width, is_screen_space=False, is_homogeneous=False)
  nic_x = np.reshape(pixel_coords[:, :, 0], [height, width])
  nic_y = np.reshape(pixel_coords[:, :, 1], [height, width])
  # GAPS nic coordinates have an origin at the center of the image, not
  # in the corner:
  nic_x = 2 * nic_x - 1.0
  nic_y = 2 * nic_y - 1.0
  nic_d = -np.reshape(im, [height, width])
  aspect = height / float(width)
  yfov = math.atan(aspect * math.tan(xfov))

  intrinsics_00 = 1.0 / math.tan(xfov)
  intrinsics_11 = 1.0 / math.tan(yfov)

  cam_x = nic_x * -nic_d / intrinsics_00
  cam_y = nic_y * nic_d / intrinsics_11
  cam_z = nic_d

  cam_xyz = np.stack([cam_x, cam_y, cam_z], axis=2)
  return cam_xyz


def apply_tx_np(samples, tx, is_point=True):
  shape_in = samples.shape
  flat_samples = np.reshape(samples, [-1, 3])
  flat_samples = to_homogenous_np(flat_samples, is_point=is_point)
  flat_samples = np.matmul(flat_samples, tx.T)
  flat_samples = flat_samples[:, :3]
  return np.reshape(flat_samples, shape_in)


def depth_image_to_sdf_constraints(im, cam2world, xfov=0.5):
  """Estimates inside/outside constraints from a gaps depth image."""
  im = _unbatch(im)
  cam2world = _unbatch(cam2world)
  height, width, _ = im.shape
  cam_xyz = depth_to_cam_np(im, xfov)
  world_xyz = apply_tx_np(cam_xyz, cam2world, is_point=True)
  ray_xyz = apply_tx_np(cam_xyz, cam2world, is_point=False)
  ray_xyz = ray_xyz / np.linalg.norm(ray_xyz, axis=-1, keepdims=True)
  delta = 0.005
  pos_constraint = world_xyz - delta * ray_xyz
  neg_constraint = world_xyz + delta * ray_xyz
  sample_shape = [height * width, 3]
  pos_constraint = np.reshape(pos_constraint, sample_shape)
  neg_constraint = np.reshape(neg_constraint, sample_shape)
  sdf_shape = [height * width, 1]
  zero = np.zeros(sdf_shape, dtype=np.float32)

  # Filter out the background
  is_valid = np.reshape(im, [-1]) != 0.0
  pos_constraint = pos_constraint[is_valid, :]
  neg_constraint = neg_constraint[is_valid, :]
  zero = zero[is_valid, :]

  samples = np.concatenate([pos_constraint, neg_constraint], axis=0)
  constraints = np.concatenate([zero + delta, zero - delta], axis=0)
  return samples, constraints


def depth_dodeca_to_sdf_constraints(depth_ims):
  """Estimates inside/outside constraints from a depth dodecahedron."""
  cam2world = np.split(get_dodeca_camera_to_worlds(), 20)
  depth_ims = np.split(_unbatch(depth_ims), 20)
  samps = []
  constraints = []
  for i in range(20):
    s, c = depth_image_to_sdf_constraints(depth_ims[i], cam2world[i])
    samps.append(s)
    constraints.append(c)
  samps = np.concatenate(samps)
  constraints = np.concatenate(constraints)
  return samps, constraints


def depth_dodeca_to_samples(dodeca):
  samples, sdf_constraints = depth_dodeca_to_sdf_constraints(dodeca)
  all_samples = np.concatenate([samples, sdf_constraints], axis=-1)
  return all_samples


def depth_image_to_class_constraints(im, cam2world, xfov=0.5):
  samples, sdf_constraints = depth_image_to_sdf_constraints(im, cam2world, xfov)
  class_constraints = sdf_constraints > 0
  return samples, class_constraints


def depth_image_to_samples(im, cam2world, xfov=0.5):  # pylint:disable=unused-argument
  """A wrapper for depth_image_to_sdf_constraints to return samples."""
  samples, sdf_constraints = depth_image_to_sdf_constraints(im, cam2world)
  all_samples = np.concatenate([samples, sdf_constraints], axis=-1)
  return all_samples


def apply_4x4(tensor, tx, are_points=True, batch_rank=None, sample_rank=None):
  """Applies a 4x4 matrix to 3D points/vectors.

  Args:
    tensor: Tensor with shape [batching_dims] + [sample_dims] + [3].
    tx: Tensor with shape [batching_dims] + [4, 4].
    are_points: Boolean. Whether to treat the samples as points or vectors.
    batch_rank: The number of leading batch dimensions. Optional, just used to
      enforce the shapes are as expected.
    sample_rank: The number of sample dimensions. Optional, just used to enforce
      the shapes are as expected.

  Returns:
    Tensor with shape [..., sample_count, 3].
  """
  expected_batch_rank = batch_rank
  expected_sample_rank = sample_rank
  batching_dims = tx.get_shape().as_list()[:-2]
  batch_rank = len(batching_dims)
  if expected_batch_rank is not None:
    assert batch_rank == expected_batch_rank
  # flat_batch_count = int(np.prod(batching_dims))

  sample_dims = tensor.get_shape().as_list()[batch_rank:-1]
  sample_rank = len(sample_dims)
  if expected_sample_rank is not None:
    assert sample_rank == expected_sample_rank
  flat_sample_count = int(np.prod(sample_dims))
  tensor = tf.ensure_shape(tensor, batching_dims + sample_dims + [3])
  tx = tf.ensure_shape(tx, batching_dims + [4, 4])
  assert sample_rank >= 1
  assert batch_rank >= 0
  if sample_rank > 1:
    tensor = tf.reshape(tensor, batching_dims + [flat_sample_count, 3])
  initializer = tf.ones if are_points else tf.zeros
  w = initializer(batching_dims + [flat_sample_count, 1], dtype=tf.float32)
  tensor = tf.concat([tensor, w], axis=-1)
  tensor = tf.matmul(tensor, tx, transpose_b=True)
  tensor = tensor[..., :3]
  if sample_rank > 1:
    tensor = tf.reshape(tensor, batching_dims + sample_dims + [3])
  return tensor


def depth_image_to_xyz_image(depth_images, world_to_camera, xfov=0.5):
  """Converts GAPS depth images to world space."""
  batch_size, height, width, channel_count = depth_images.get_shape().as_list()
  assert channel_count == 1

  camera_to_world_mat = tf.matrix_inverse(world_to_camera)

  pixel_coords = np_util.make_coordinate_grid(
      height, width, is_screen_space=False, is_homogeneous=False)
  nic_x = np.tile(
      np.reshape(pixel_coords[:, :, 0], [1, height, width]), [batch_size, 1, 1])
  nic_y = np.tile(
      np.reshape(pixel_coords[:, :, 1], [1, height, width]), [batch_size, 1, 1])

  nic_x = 2 * nic_x - 1.0
  nic_y = 2 * nic_y - 1.0
  nic_d = -tf.reshape(depth_images, [batch_size, height, width])

  aspect = height / float(width)
  yfov = math.atan(aspect * math.tan(xfov))

  intrinsics_00 = 1.0 / math.tan(xfov)
  intrinsics_11 = 1.0 / math.tan(yfov)

  nic_xyz = tf.stack([nic_x, nic_y, nic_d], axis=3)
  flat_nic_xyz = tf.reshape(nic_xyz, [batch_size, height * width, 3])

  camera_x = (nic_x) * -nic_d / intrinsics_00
  camera_y = (nic_y) * nic_d / intrinsics_11
  camera_z = nic_d
  homogeneous_coord = tf.ones_like(camera_z)
  camera_xyz = tf.stack([camera_x, camera_y, camera_z, homogeneous_coord],
                        axis=3)
  flat_camera_xyzw = tf.reshape(camera_xyz, [batch_size, height * width, 4])
  flat_world_xyz = tf.matmul(
      flat_camera_xyzw, camera_to_world_mat, transpose_b=True)
  world_xyz = tf.reshape(flat_world_xyz, [batch_size, height, width, 4])
  world_xyz = world_xyz[:, :, :, :3]
  return world_xyz, flat_camera_xyzw[:, :, :3], flat_nic_xyz


def interpolate_from_grid_coordinates(samples, grid):
  """Performs trilinear interpolation to estimate the value of a grid function.

  This function makes several assumptions to do the lookup:
  1) The grid is LHW and has evenly spaced samples in the range (0, 1), which
    is really the screen space range [0.5, {L, H, W}-0.5].

  Args:
    samples: Tensor with shape [batch_size, sample_count, 3].
    grid: Tensor with shape [batch_size, length, height, width, 1].

  Returns:
    sample: Tensor with shape [batch_size, sample_count, 1] and type float32.
    mask: Tensor with shape [batch_size, sample_count, 1] and type float32
  """
  batch_size, length, height, width = grid.get_shape().as_list()[:4]
  # These asserts aren't required by the algorithm, but they are currently
  # true for the pipeline:
  assert length == height
  assert length == width
  sample_count = samples.get_shape().as_list()[1]
  tf_util.assert_shape(samples, [batch_size, sample_count, 3],
                       'interpolate_from_grid:samples')
  tf_util.assert_shape(grid, [batch_size, length, height, width, 1],
                       'interpolate_from_grid:grid')
  offset_samples = samples  # Used to subtract 0.5
  lower_coords = tf.cast(tf.math.floor(offset_samples), dtype=tf.int32)
  upper_coords = lower_coords + 1
  alphas = tf.floormod(offset_samples, 1.0)

  maximum_value = grid.get_shape().as_list()[1:4]
  size_per_channel = tf.tile(
      tf.reshape(tf.constant(maximum_value, dtype=tf.int32), [1, 1, 3]),
      [batch_size, sample_count, 1])
  # We only need to check that the floor is at least zero and the ceil is
  # no greater than the max index, because floor round negative numbers to
  # be more negative:
  is_valid = tf.logical_and(lower_coords >= 0, upper_coords < size_per_channel)
  # Validity mask has shape [batch_size, sample_count] and is 1.0 where all of
  # x,y,z are within the [0,1] range of the grid.
  validity_mask = tf.reduce_min(
      tf.cast(is_valid, dtype=tf.float32), axis=2, keep_dims=True)

  lookup_coords = [[[], []], [[], []]]
  corners = [[[], []], [[], []]]
  flattened_grid = tf.reshape(grid, [batch_size, length * height * width])
  for xi, x_coord in enumerate([lower_coords[:, :, 0], upper_coords[:, :, 0]]):
    x_coord = tf.clip_by_value(x_coord, 0, width - 1)
    for yi, y_coord in enumerate([lower_coords[:, :, 1], upper_coords[:, :,
                                                                      1]]):
      y_coord = tf.clip_by_value(y_coord, 0, height - 1)
      for zi, z_coord in enumerate(
          [lower_coords[:, :, 2], upper_coords[:, :, 2]]):
        z_coord = tf.clip_by_value(z_coord, 0, length - 1)
        flat_lookup = z_coord * height * width + y_coord * width + x_coord
        lookup_coords[xi][yi].append(flat_lookup)
        lookup_result = tf.batch_gather(flattened_grid, flat_lookup)
        tf_util.assert_shape(lookup_result, [batch_size, sample_count],
                             'interpolate_from_grid:lookup_result x/8')
        print_op = tf.print('corner xyz=%i, %i, %i' % (xi, yi, zi),
                            lookup_result, '\n', 'flat_lookup:', flat_lookup,
                            '\n\n')
        with tf.control_dependencies([print_op]):
          lookup_result = 1.0 * lookup_result
        corners[xi][yi].append(lookup_result)

  alpha_x, alpha_y, alpha_z = tf.unstack(alphas, axis=2)
  one_minus_alpha_x = 1.0 - alpha_x
  one_minus_alpha_y = 1.0 - alpha_y
  # First interpolate a face along x:
  f00 = corners[0][0][0] * one_minus_alpha_x + corners[1][0][0] * alpha_x
  f01 = corners[0][0][1] * one_minus_alpha_x + corners[1][0][1] * alpha_x
  f10 = corners[0][1][0] * one_minus_alpha_x + corners[1][1][0] * alpha_x
  f11 = corners[0][1][1] * one_minus_alpha_x + corners[1][1][1] * alpha_x
  # Next interpolate a long along y:
  l0 = f00 * one_minus_alpha_y + f10 * alpha_y
  l1 = f01 * one_minus_alpha_y + f11 * alpha_y

  # Finally interpolate a point along z:
  p = l0 * (1.0 - alpha_z) + l1 * alpha_z

  tf_util.assert_shape(p, [batch_size, sample_count], 'interpolate_from_grid:p')

  p = tf.reshape(p, [batch_size, sample_count, 1])
  validity_mask = tf.reshape(validity_mask, [batch_size, sample_count, 1])
  return p, validity_mask

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
"""Numpy utility functions."""

# Could penalize minimum volume for RBFs.
# Could penalize heavily if another quadric has higher weight within this quad's
# radius.
# Plot fraction in which 1 quadrics has 90% or more of the total weight.
# If total weight is below epsilon, don't show (or show which those are). Set
# this using a colab notebook, which will need to work first...
# Replace exp with some better falloff function!

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
# This import pulls in necessary backend dependencies for matplotlib:
# pylint:disable=unused-import
from mpl_toolkits.mplot3d import Axes3D
# pylint:enable=unused-import
import numpy as np
from skimage import measure
from skimage.transform import resize

import tensorflow as tf


def batch_np(arr, batch_size):
  s = arr.shape
  arr = np.expand_dims(arr, 0)
  tile = [batch_size] + [1] * len(s)
  return np.tile(arr, tile)


def make_coordinate_grid(height, width, is_screen_space, is_homogeneous):
  """Returns an array containing the coordinate grid values for an image.

  Outputs a numpy array to avoid adding unnecessary operations to the graph.

  Args:
    height: int containing the height of the output image.
    width: int containing the width of the output image.
    is_screen_space: bool. If true, then the coordinates are measured in pixels.
      If false, they are in the range (0-1).
    is_homogeneous: bool. If true, then a 1 is appended to the end of each
      coordinate.

  Returns:
    coords: numpy array of shape [height, width, 2] or [height, width, 3],
      depending on whether is_homogeneous is true. The value at location
      [i, j, :] is the (x,y) or (x,y,1) coordinate value at that location.
  """
  with tf.name_scope('make_coordinate_grid'):
    x_coords = np.linspace(0.5, width - 0.5, width)
    y_coords = np.linspace(0.5, height - 0.5, height)
    if not is_screen_space:
      x_coords /= width
      y_coords /= height
    grid_x, grid_y = np.meshgrid(
        x_coords, y_coords, sparse=False, indexing='xy')
    if is_homogeneous:
      homogeneous_coords = np.ones_like(grid_x)
      return np.stack([grid_x, grid_y, homogeneous_coords], axis=2)
    return np.stack([grid_x, grid_y], axis=2)


def make_coordinate_grid_3d(length, height, width, is_screen_space,
                            is_homogeneous):
  """Returns an array containing the coordinate grid values for a volume.

  Outputs a numpy array to avoid adding unnecessary operations to the graph.

  Args:
    length: int containing the length of the output volume.
    height: int containing the height of the output volume.
    width: int containing the width of the output volume.
    is_screen_space: bool. If true, then the coordinates are measured in pixels.
      If false, they are in the range (0-1).
    is_homogeneous: bool. If true, then a 1 is appended to the end of each
      coordinate.

  Returns:
    coords: numpy array of shape [length, height, width, 3] or
      [length, height, width, 4], depending on whether is_homogeneous is true.
      The value at location [i, j, k, :] is the (x,y,z) or (x,y,z,1) coordinate
      value at that location.
  """
  x_coords = np.linspace(0.5, width - 0.5, width)
  y_coords = np.linspace(0.5, height - 0.5, height)
  z_coords = np.linspace(0.5, length - 0.5, length)
  if not is_screen_space:
    x_coords /= width
    y_coords /= height
    z_coords /= length
  grid_x, grid_y, grid_z = np.meshgrid(
      x_coords, y_coords, z_coords, sparse=False, indexing='ij')
  if is_homogeneous:
    homogeneous_coords = np.ones_like(grid_x)
    grid = np.stack([grid_x, grid_y, grid_z, homogeneous_coords], axis=3)
  else:
    grid = np.stack([grid_x, grid_y, grid_z], axis=3)
  # Currently the order is (w, h, l), but we need (l, h, w) for
  # TensorFlow compatibility:
  return np.swapaxes(grid, 0, 2)


def filter_valid(mask, vals):
  """Filters an array to only its valid values.

  Args:
    mask: Boolean numpy array with shape [...] or [..., 1].
    vals: Numpy array with shape [...] or [..., 1]. Trailing 1 doesn't have to
      match.

  Returns:
    Numpy array with the same rank as vals but without elements that have a
    False value in the mask.
  """
  if mask.shape[-1] == 1:
    assert len(mask.shape) == len(vals.shape)
    mask = np.reshape(mask, mask.shape[:-1])
  else:
    assert len(mask.shape) == len(vals.shape) - 1
  return vals[mask, :]


def zero_by_mask(mask, vals, replace_with=0.0):
  """Sets the invalid part of vals to the value of replace_with.

  Args:
    mask: Boolean array that matches vals in shape, except for squeezable dims
      and the final dimension (the 'channel' dimension).
    vals: Numpy array with shape [..., channel_count].
    replace_with: Value to put in invalid locations, if not 0.0. Dtype should be
      compatible with that of vals.

  Returns:
    Numpy array with shape [..., channel_count] with 0 in invalid locations.
  """
  vals = vals.copy()
  mask = np.reshape(mask, vals.shape[:-1])
  vals[np.logical_not(mask), :] = replace_with
  return vals


def make_mask(im, thresh=0.0):
  """Computes a numpy boolean mask from an array of (nonnegative) floats."""
  mv = np.min(im)
  assert mv >= 0.0
  return im > thresh


def make_pixel_mask(im):
  """Computes a (height, width) mask that is true when any channel is true."""
  channels_valid = im.astype(np.bool)
  mask = np.any(channels_valid, axis=2)
  assert len(mask.shape) == 2
  return mask


def thresh_and_radius_to_distance(radius, thresh):
  """Computes the distance at which an rbf reaches a value."""
  # Given a radius in world units, and a threshold, computes the distance in
  # world units at which an rbf with that radius would reach a value of thresh
  return np.sqrt(-2.0 * radius * np.log(thresh))


def plot_rbfs_at_thresh(centers, radii, thresh=0.5):
  """Generates images visualizing the rbfs at a threshold.

  Args:
    centers: numpy array with shape (batch_size, rbf_count, 3).
    radii: numpy array with shape (batch_size, rbf_count, 3 or 1).
    thresh: The threshold at which to show RBFs.

  Returns:
    Visualization images.
  """
  batch_size, rbf_count = centers.shape[0:2]
  outputs = []
  if radii.shape[2] == 1:
    radii = np.tile(radii, [1, 1, 3])

  # Set of all spherical angles:
  u = np.linspace(0, 2 * np.pi, 100)
  v = np.linspace(0, np.pi, 100)

  unit_sphere_x = np.outer(np.cos(u), np.sin(v))
  unit_sphere_y = np.outer(np.sin(u), np.sin(v))
  unit_sphere_z = np.outer(np.ones_like(u), np.cos(v))

  for bi in range(batch_size):
    fig = plt.figure(figsize=4 * plt.figaspect(1))  # Square figure
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')
    for rbf_idx in range(rbf_count):

      # Radii corresponding to the coefficients:
      rx = thresh_and_radius_to_distance(radii[bi, rbf_idx, 0], thresh)
      ry = thresh_and_radius_to_distance(radii[bi, rbf_idx, 1], thresh)
      rz = thresh_and_radius_to_distance(radii[bi, rbf_idx, 2], thresh)
      center_x = centers[bi, rbf_idx, 0]
      center_y = centers[bi, rbf_idx, 1]
      center_z = centers[bi, rbf_idx, 2]

      # Cartesian coordinates that correspond to the spherical angles:
      # (this is the equation of an ellipsoid):
      x = rx * unit_sphere_x + center_x
      y = ry * unit_sphere_y + center_y
      z = rz * unit_sphere_z + center_z

      # Plot:
      ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b')

    # Adjustment of the axes, so that they all have the same span:
    for axis in 'xyz':
      getattr(ax, 'set_%slim' % axis)((-0.5, 0.5))

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    height, width = int(height), int(width)
    image = np.fromstring(
        canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    outputs.append(image)
    plt.close('all')
  return np.stack(outputs)


def plot_rbfs(centers, radii, scale=10.0):
  """Generates images visualizing the rbfs.

  Args:
    centers: numpy array with shape (batch_size, rbf_count, 3).
    radii: numpy array with shape (batch_size, rbf_count, 3 or 1).
    scale: The multiplication factor in radii to visualize.

  Returns:
    np array with the visualization images.
  """
  batch_size, rbf_count = centers.shape[0:2]
  outputs = []
  if radii.shape[2] == 1:
    radii = np.tile(radii, [1, 1, 3])

  # Set of all spherical angles:
  u = np.linspace(0, 2 * np.pi, 100)
  v = np.linspace(0, np.pi, 100)

  unit_sphere_x = np.outer(np.cos(u), np.sin(v))
  unit_sphere_y = np.outer(np.sin(u), np.sin(v))
  unit_sphere_z = np.outer(np.ones_like(u), np.cos(v))

  for bi in range(batch_size):
    fig = plt.figure(figsize=4 * plt.figaspect(1))  # Square figure
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')
    for rbf_idx in range(rbf_count):

      # Radii corresponding to the coefficients:
      rx = radii[bi, rbf_idx, 0] * scale
      ry = radii[bi, rbf_idx, 1] * scale
      rz = radii[bi, rbf_idx, 2] * scale
      center_x = centers[bi, rbf_idx, 0]
      center_y = centers[bi, rbf_idx, 1]
      center_z = centers[bi, rbf_idx, 2]

      # Cartesian coordinates that correspond to the spherical angles:
      # (this is the equation of an ellipsoid):
      x = rx * unit_sphere_x + center_x
      y = ry * unit_sphere_y + center_y
      z = rz * unit_sphere_z + center_z

      # Plot:
      ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b')

    # Adjustment of the axes, so that they all have the same span:
    for axis in 'xyz':
      getattr(ax, 'set_%slim' % axis)((-0.5, 0.5))

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    height, width = int(height), int(width)
    image = np.fromstring(
        canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    outputs.append(image)
    plt.close('all')
  return np.stack(outputs)


def cube_and_render(volume, thresh):
  """Extract a mesh and render an image."""
  volume = np.squeeze(volume)
  length, height, width = volume.shape
  resolution = length
  # This function doesn't support non-cube volumes:
  assert resolution == height and resolution == width
  try:
    vertices, faces, _, _ = measure.marching_cubes_lewiner(volume, thresh)
    x, y, z = [np.array(x) for x in zip(*vertices)]
    xyzw = np.stack([x, y, z, np.ones_like(x)], axis=1)
    # Center the volume around the origin:
    xyzw += np.array(
        [[-resolution / 2.0, -resolution / 2.0, -resolution / 2.0, 0.]])
    # This assumes the world is right handed with y up; matplotlib's renderer
    # has z up and is left handed:
    # Reflect across z, rotate about x, and rescale to [-0.5, 0.5].
    xyzw *= np.array(
        [[1.0 / resolution, 1.0 / resolution, -1.0 / resolution, 1]])
    y_up_to_z_up = np.array([[1., 0., 0., 0.], [0., 0., -1., 0.],
                             [0., 1., 0., 0.], [0., 0., 0., 1.]])
    xyzw = np.matmul(y_up_to_z_up, xyzw.T).T

    world_space_xyz = np.copy(xyzw[:, :3])

    # TODO(kgenova): Apply any transformation you want to the mesh in the same
    # way as above (i.e. the inverse of the camera extrinsics). If that's hard
    # to express in this space it could go before the transformations above.
    # This is an example that just rotates the object off axis a bit.
    rotation_around_z = np.array([[-0.9396926, -0.3420202, 0., 0.],
                                  [0.3420202, -0.9396926, 0., 0.],
                                  [0., 0., 1.0, 0.], [0., 0., 0., 1.]])
    xyzw = np.matmul(rotation_around_z, xyzw.T).T

    # Back to matplotlib format:
    x, y, z = [np.squeeze(x) for x in np.split(xyzw[:, :3], 3, axis=1)]
    fig = plt.figure(figsize=(8, 8))
    dpi = fig.get_dpi()
    fig.set_size_inches(1220.0 / float(dpi), 1220.0 / float(dpi))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(
        x,
        y,
        z,
        triangles=faces,
        linewidth=0.0,
        shade=True,
        cmap='viridis',
        antialiased=False)
    ax.set_xlim(-0.35, 0.35)
    ax.set_ylim(-0.35, 0.35)
    ax.set_zlim(-0.45, 0.25)
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    height, width = int(height), int(width)
    image = np.fromstring(
        canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
  except ValueError:
    image = np.zeros([128, 128, 3], dtype=np.uint8) + 255
    world_space_xyz = np.zeros([100, 3])
    faces = np.zeros([300, 3])
  except RuntimeError:
    image = np.zeros([128, 128, 3], dtype=np.uint8) + 255
    world_space_xyz = np.zeros([100, 3])
    faces = np.zeros([300, 3])
  plt.close('all')
  return image, world_space_xyz, np.copy(faces)


def sample_surface(quadrics, centers, radii, length, height, width,
                   renormalize):
  """Deprecated: Samples the SIF value at the surface locations."""
  quadric_count = quadrics.shape[0]
  homogeneous_coords = make_coordinate_grid_3d(
      length, height, width, is_screen_space=False, is_homogeneous=True)
  homogeneous_coords = np.reshape(homogeneous_coords,
                                  [length, height, width, 4])
  homogeneous_coords[:, :, :, :3] -= 0.5
  flat_coords = np.reshape(homogeneous_coords, [length * height * width, 4])

  surface_volume = np.zeros([length, height, width, 1], dtype=np.float32)

  max_bf_weights = np.zeros([length, height, width, 1], dtype=np.float32)
  total_bf_weights = np.zeros([length, height, width, 1], dtype=np.float32)
  for qi in range(quadric_count):
    quadric = quadrics[qi, :, :]
    center = centers[qi, :]
    # This is the one to uncomment when updating the renderer.
    radius = radii[qi, :3]
    offset_coords = flat_coords.copy()
    offset_coords[:, :3] -= np.reshape(center, [1, 3])
    half_distance = np.matmul(quadric, offset_coords.T).T
    algebraic_distance = np.sum(offset_coords * half_distance, axis=1)

    squared_diff = (offset_coords[:, :3] * offset_coords[:, :3])
    scale = np.reciprocal(np.minimum(-2 * radius, 1e-6))
    bf_weights = np.exp(np.sum(scale * squared_diff, axis=1))
    volume_addition = np.reshape(algebraic_distance * bf_weights,
                                 [length, height, width, 1])
    max_bf_weights = np.maximum(
        np.reshape(bf_weights, [length, height, width, 1]), max_bf_weights)
    total_bf_weights += np.reshape(bf_weights, [length, height, width, 1])
    surface_volume += volume_addition
  if renormalize:
    surface_volume /= total_bf_weights
  surface_volume[max_bf_weights < 0.0001] = 1.0
  return surface_volume


def visualize_prediction(quadrics,
                         centers,
                         radii,
                         renormalize,
                         thresh=0.0,
                         input_volumes=None):
  """Creates a [batch_size, height, width, 3/4] image visualizing the output."""
  # TODO(kgenova) All of this needs to go or be rewritten to work with the
  # diffren sampler.
  prediction_count = quadrics.shape[0]
  images = []
  volumes = []
  for i in range(prediction_count):
    if input_volumes is not None:
      volume = input_volumes[i, ...]
    else:
      volume = sample_surface(
          quadrics[i, :, :, :],
          centers[i, :, :],
          radii[i, :, :],
          length=64,
          height=64,
          width=64,
          renormalize=renormalize)
    image, _, _ = cube_and_render(volume, thresh)
    target_height = 256
    target_width = 256
    image = resize(image, (target_width, target_height))
    # Append opacity:
    image = np.pad(
        image, [[0, 0], [0, 0], [0, 1]], mode='constant', constant_values=1.0)
    images.append(image)
    volumes.append(volume)
  return np.stack(images), np.stack(volumes)

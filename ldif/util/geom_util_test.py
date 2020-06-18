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
"""Tests for ldif.util.geom_util."""

import os

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import googletest

# ldif is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import file_util
from ldif.util import geom_util
from ldif.util import path_util
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order

DISTANCE_EPS = 1e-6


def write_points(points, fname, validity=None):
  output_dir = path_util.create_test_output_dir()
  output_fname = os.path.join(output_dir, fname)

  points = np.reshape(points, [-1, 3])
  if validity is not None:
    validity = np.reshape(validity, [-1, 1])
    log.info(f'Points, validity shape: {points.shape}, {validity.shape}')
    assert validity.shape[0] == points.shape[0]
    validity = np.tile(validity, [1, 3])
    valid_points = points[validity]
    points = np.reshape(valid_points, [-1, 3])

  normals = np.zeros_like(points)
  log.info(f'Points shape: {points.shape}')
  log.info(f'Normals shape: {normals.shape}')
  np.savetxt(output_fname + '.pts', np.concatenate([points, normals], axis=1))


def validate_zero_set_interpolation(zero_set_points, target_volume,
                                    model_config):
  basic_interp, basic_validity_mask = geom_util.interpolate_from_grid(
      zero_set_points, target_volume)
  mean_interp = tf.reduce_mean(basic_interp)
  mean_abs_interp = tf.reduce_mean(tf.abs(basic_interp))
  mean_abs_valid_interp = tf.reduce_mean(
      tf.abs(basic_validity_mask * basic_interp))

  dataset_split = model_config.inputs['split']
  tf.summary.scalar('%s-sanity-check/mean_interp' % dataset_split, mean_interp)
  tf.summary.scalar('%s-sanity-check/mean_abs_interp' % dataset_split,
                    mean_abs_interp)
  tf.summary.scalar('%s-sanity-check/mean_abs_valid_interp' % dataset_split,
                    mean_abs_valid_interp)
  seq = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1],
         [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]]
  for sign_flips in seq:
    points = zero_set_points
    xf, yf, zf = sign_flips
    name = '%s%s%s' % (str(xf), str(yf), str(zf))
    xp, yp, zp = tf.split(points, [1, 1, 1], axis=-1)
    if xf == -1:
      xp = 1 - xp
    if yf == -1:
      yp = 1 - yp
    if zf == -1:
      zp = 1 - zp
    points = tf.concat([xp, yp, zp], axis=-1)
    flipped_interp = geom_util.interpolate_from_grid(points, target_volume)
    mean_interp = tf.reduce_mean(flipped_interp)
    mean_abs_interp = tf.reduce_mean(tf.abs(flipped_interp))
    if xf == 1 and yf == 1 and zf == 1:
      point_dif = tf.reduce_mean(tf.abs(points - zero_set_points))
      tf.summary.scalar('%s-sanity-check/point_MAD' % dataset_split, point_dif)
      result_dif = tf.reduce_mean(tf.abs(flipped_interp - basic_interp))
      tf.summary.scalar('%s-sanity-check/result_MAD' % dataset_split,
                        result_dif)
    tf.summary.scalar('%s-sanity-check/mean-interp_%s' % (dataset_split, name),
                      mean_interp)
    tf.summary.scalar(
        '%s-sanity-check/mean-abs-interp_%s' % (dataset_split, name),
        mean_abs_interp)


class GeomUtilTest(tf.test.TestCase):

  def setUp(self):
    super(GeomUtilTest, self).setUp()

    self.test_data_directory = path_util.util_test_data_path()

  def _testInterpolateFromGrid(self):
    """Tests trilinear interpolation."""
    grid = np.zeros([2, 2, 2], dtype=np.float32)
    grid[0, 0, 0] = 0
    grid[0, 0, 1] = 1
    grid[0, 1, 0] = 1
    grid[0, 1, 1] = 0
    grid[1, 0, 0] = 0
    grid[1, 0, 1] = 1
    grid[1, 1, 0] = 0
    grid[1, 1, 1] = 1
    grid = np.reshape(grid, [1, 2, 2, 2, 1])
    grid_tf = tf.constant(grid, dtype=tf.float32)
    samples = tf.constant([[[0.5, 0.5, 0.5]]], dtype=tf.float32)
    samples = tf.reshape(samples, [1, 1, 3])
    result_tf = geom_util.interpolate_from_grid(samples, grid_tf)
    expected = np.array([0.5])

    with self.test_session() as sess:
      returned, validity = sess.run(result_tf)
    log.info(f'result {returned}')
    log.info(f'validity: {validity}')
    distance = float(np.reshape(np.abs(expected - returned), [1]))
    self.assertLess(
        distance, DISTANCE_EPS, 'Expected \n%s\n but got \n%s' %
        (np.array_str(expected), np.array_str(returned)))

  def _testInterpolateFromGrid34(self):
    """Tests trilinear interpolation."""
    grid = np.zeros([2, 2, 2], dtype=np.float32)
    grid[0, 0, 0] = 0
    grid[0, 0, 1] = 0
    grid[0, 1, 0] = 0
    grid[0, 1, 1] = 0
    grid[1, 0, 0] = 1
    grid[1, 0, 1] = 1
    grid[1, 1, 0] = 1
    grid[1, 1, 1] = 1
    grid = np.reshape(grid, [1, 2, 2, 2, 1])
    grid_tf = tf.constant(grid, dtype=tf.float32)
    samples = tf.constant([[[0.5, 0.5, 0.375]]], dtype=tf.float32)
    samples = tf.reshape(samples, [1, 1, 3])
    result_tf = geom_util.interpolate_from_grid(samples, grid_tf)
    expected = np.array([0.25])

    with self.test_session() as sess:
      returned, validity = sess.run(result_tf)
    log.info(f'result {returned}')
    log.info(f'validity: {validity}')
    distance = float(np.reshape(np.abs(expected - returned), [1]))
    self.assertLess(
        distance, DISTANCE_EPS, 'Expected \n%s\n but got \n%s' %
        (np.array_str(expected), np.array_str(returned)))

  def testInterpolateFromGrid_Real_SDF(self):
    grid_filename = '1042d723dfc31ce5ec56aed2da084563_kd_2_sdf.npy'
    samples_filename = '1042d723dfc31ce5ec56aed2da084563_pts.npy'
    tx_filename = '1042d723dfc31ce5ec56aed2da084563_kd_2_tx.npy'

    grid_filename = os.path.join(self.test_data_directory, grid_filename)
    samples_filename = os.path.join(self.test_data_directory, samples_filename)
    tx_filename = os.path.join(self.test_data_directory, tx_filename)

    grid = np.reshape(np.load(grid_filename), [1, 64, 64, 64, 1])
    samples = np.reshape(np.load(samples_filename)[:, 0:3], [1, 100000, 3])
    sample_count = 10
    samples = samples[:, :sample_count, :]

    np.random.seed(0)
    np.random.shuffle(samples)
    tx = np.reshape(np.load(tx_filename), [4, 4])
    homogeneous_coord = np.ones([1, sample_count, 1], dtype=np.float32)
    homogeneous_samples = np.concatenate([samples, homogeneous_coord], axis=2)
    samples_to_tx = np.reshape(homogeneous_samples, [sample_count, 4]).T
    toms_transformed_samples = np.matmul(tx, samples_to_tx).T
    toms_transformed_samples = np.reshape(toms_transformed_samples[:, :3],
                                          [1, sample_count, 3])

    my_transformed_samples = (samples + 0.5) * 63.0
    samples = my_transformed_samples  # toms_transformed_samples

    samples_tf = tf.constant(samples, dtype=tf.float32)
    grid_tf = tf.constant(grid, dtype=tf.float32)

    interpolated, validity_mask = geom_util.interpolate_from_grid_coordinates(
        samples_tf, grid_tf)
    with self.test_session() as sess:
      interpolated_np, validity_mask_np = sess.run(
          [interpolated, validity_mask], feed_dict={})
      del interpolated_np
      del validity_mask_np

  def testInterpolateCleanedSDF(self):
    grid_filename = '541e331334c95e5a3d2617f9171b5ccb_cleaned_sdf.npy'
    grid_filename = os.path.join(self.test_data_directory, grid_filename)

    surface_points_filename = '541e331334c95e5a3d2617f9171b5ccb_surface_points.npy'
    surface_points_filename = os.path.join(self.test_data_directory,
                                           surface_points_filename)
    surface_points = np.reshape(
        np.load(surface_points_filename), [1, 100000, 6])
    surface_points = tf.constant(surface_points[:, ::5000, :3])
    grid = tf.constant(np.reshape(np.load(grid_filename), [1, 64, 64, 64, 1]))
    interpolated_sdfs_tf, validity_mask_tf = geom_util.interpolate_from_grid(
        surface_points, grid)

    with self.test_session() as sess:
      interpolated_sdfs_np, validity_mask_np, surface_points_np = sess.run(
          [interpolated_sdfs_tf, validity_mask_tf, surface_points])
      log.info('Mean abs interpolated clean surface point: '
               f'{np.mean(np.abs(interpolated_sdfs_np))}')
      log.info(f'surface_point interpolated_sdfs: {interpolated_sdfs_np}')
      log.info(f'surface_point validity_mask: {validity_mask_np}')
      log.info(f'surface_point surface xyz: {surface_points_np}')

  def testBatchedDodecaDepthToXYZ(self):
    npz_fnames = [
        '53502c15c30d06496e4e9b7d32a4f90d.npz',
        '53a60f2eb6b1a3e9afe717997470b28d.npz'
    ]
    all_depth_ims = []
    for fname in npz_fnames:
      npz_filename = os.path.join(self.test_data_directory, fname)
      all_depth_ims.append(
          file_util.read_npz(npz_filename)['gaps_depth'] / 1000.0)
    depth_ims_tf = tf.stack(all_depth_ims)
    xyz_tf = geom_util.transform_depth_dodeca_to_xyz_dodeca(depth_ims_tf)
    with self.test_session() as session:
      xyz_np = session.run(xyz_tf)
      output_dir = path_util.create_test_output_dir()
      for i in range(len(npz_fnames)):
        np.save(output_dir + '/xyz_im_%i.npy' % i, xyz_np[i, ...])

  def testAllToXYZ(self):
    npz_filename = '53a60f2eb6b1a3e9afe717997470b28d.npz'
    npz_filename = os.path.join(self.test_data_directory, npz_filename)
    npz = file_util.read_npz(npz_filename)
    depth_ims = npz['gaps_depth'] / 1000.0
    surface_pts = npz['surface_points']
    depth_ims_tf = tf.reshape(
        tf.constant(depth_ims, dtype=tf.float32), [1, 20, 224, 224, 1])
    xyz_images = geom_util.transform_depth_dodeca_to_xyz_dodeca(depth_ims_tf)
    with self.test_session() as session:
      xyz_np = session.run(xyz_images)
      output_dir = path_util.create_test_output_dir()
      valid_locations = depth_ims != 0.0
      valid_locations = np.reshape(valid_locations, [1, 20, 224, 224])
      filtered_xyz = xyz_np[valid_locations, :]
      xyz_from_depth = np.reshape(filtered_xyz, [-1, 3])
      np.random.shuffle(xyz_from_depth)
      xyz_from_depth = xyz_from_depth[:10000, :]
      np.save(output_dir + '/surface_pts.npy', surface_pts)
      np.save(output_dir + '/depth_xyz.npy', xyz_np)
      np.save(output_dir + '/depth.npy', depth_ims)
      np.save(output_dir + '/filtered_xyz.npy', filtered_xyz)
      np.save(output_dir + '/xyz_from_depth.npy', xyz_from_depth)

  def testDepthImageToXYZImage(self):
    grid_filename = '541e331334c95e5a3d2617f9171b5ccb_cleaned_sdf.npy'
    depth_filename = '541e331334c95e5a3d2617f9171b5ccb_000001_depth.png'
    grid_filename = os.path.join(self.test_data_directory, grid_filename)
    depth_filename = os.path.join(self.test_data_directory, depth_filename)

    grid = tf.constant(np.reshape(np.load(grid_filename), [1, 64, 64, 64, 1]))
    depth_image = file_util.read_image(depth_filename) / 1000.0
    height, width = depth_image.shape[0:2]
    depth_image = np.reshape(depth_image, [1, height, width, 1])
    chosen_depth_image_np = depth_image
    depth_image = tf.constant(depth_image)

    # From bkp; why are some different?
    # We have verified that this correctly specifies a camera at the viewpoint
    # and looking towards the origin; it is a distance of 1.4 from the world
    # origin.
    viewpoint = np.array([1.03276, 0.757946, -0.564739])
    towards = np.array([-0.737684, -0.54139, 0.403385])  #  = v/-1.4
    up = np.array([-0.47501, 0.840771, 0.259748])

    # Alternative:
    # viewpoint = np.array([1.087, 0.757946, -0.451614])
    # towards = np.array([-0.776426, -0.54139, 0.322582])
    # up = np.array([-0.499957, 0.840771, 0.207717])

    # The up vector is what world up, (0,1,0), the y axis maps to
    # The right vector is what the x axis maps to.
    # The towards is what the negative z axis maps to (?)
    towards = towards / np.linalg.norm(towards)
    right = np.cross(towards, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, towards)
    up = up / np.linalg.norm(up)

    # Since towards is what the -z axis maps to, -towards is what the z axis
    # maps to. So the following rotation matrix maps world XYZ axes to camera
    # XYZ axes.
    rotation = np.stack([right, up, -towards], axis=1)
    rotation_4x4 = np.eye(4)
    rotation_4x4[:3, :3] = rotation  # Takes the world and rotates it to camera
    # space.
    camera_to_world = rotation_4x4.copy()
    camera_to_world[:3, 3] = viewpoint
    camera_to_world = tf.constant(camera_to_world.astype(np.float32))
    world_to_camera = tf.reshape(tf.matrix_inverse(camera_to_world), [1, 4, 4])

    xyz_locations, flat_camera_xyz, flat_nic_xyz = geom_util.depth_image_to_xyz_image(
        depth_image, world_to_camera, xfov=0.5)
    flat_xyz_locations = tf.reshape(xyz_locations, [1, height * width, 3])
    interpolated_sdfs_tf, validity_mask_tf = geom_util.interpolate_from_grid(
        flat_xyz_locations, grid)  # Right to a least mean-abs 0.002.
    interpolated_sdfs_tf = tf.reshape(interpolated_sdfs_tf, [height, width, 1])
    nonzero_depth_mask = tf.reshape(depth_image, [height, width, 1]) > 0.05

    surface_points_filename = '541e331334c95e5a3d2617f9171b5ccb_surface_points.npy'
    surface_points_filename = os.path.join(self.test_data_directory,
                                           surface_points_filename)
    surface_points = np.reshape(
        np.load(surface_points_filename), [1, 100000, 6])
    all_surface_points_np = surface_points.copy()
    surface_points_np = surface_points[:, ::10, :3]
    surface_points_np[:, 0, :] = [0, 0, 0]  # See where the origin maps:
    surface_points = tf.constant(surface_points_np)
    surface_camera_space = tf.matmul(
        tf.pad(surface_points, [[0, 0], [0, 0], [0, 1]], constant_values=1),
        world_to_camera,
        transpose_b=True)[:, :, :3]

    with self.test_session() as sess:
      (interpolated_sdfs_np, nonzero_depth_mask_np, flat_xyz_locations_np,
       validity_mask_np, world_to_camera_np, surface_camera_space_np,
       reprojected_camera_np, flat_nic_xyz_np) = sess.run([
           interpolated_sdfs_tf, nonzero_depth_mask, flat_xyz_locations,
           validity_mask_tf, world_to_camera, surface_camera_space,
           flat_camera_xyz, flat_nic_xyz
       ])
      log.info(f'flat_xyz_locations: {flat_xyz_locations_np}')
      log.info(f'chosen_depth_image_np: {chosen_depth_image_np}')
      log.info(f'Interpolated_sdfs_np: {interpolated_sdfs_np}')
      log.info(f'validity_mask_np: {validity_mask_np}')
      log.info(f'interpolated at nonzero: '
               f'{interpolated_sdfs_np[nonzero_depth_mask_np]}')
      log.info('Mean abs interpolated value: '
               f'{np.mean(np.abs(interpolated_sdfs_np))}')
      if np.sum(nonzero_depth_mask_np) > 1e-8:
        log.info(
            'Avg abs zD interpolated value: '
            f'{np.average(np.abs(interpolated_sdfs_np), weights=nonzero_depth_mask_np)}'
        )
      log.info(f'world2cam: {world_to_camera_np}')
      log.info(f'surface_camera_space: {surface_camera_space_np}')
      log.info(f'surface_world_space: {surface_points_np}')

      distance = np.linalg.norm(
          surface_camera_space_np - surface_points_np, axis=2)
      log.info(f'surface world->cam distance: {distance}')

      output_dir = path_util.create_test_output_dir()
      write_points(flat_xyz_locations_np,
                   'valid_reprojected_depth_map_world_space',
                   nonzero_depth_mask_np)
      write_points(surface_points_np, 'chosen_surface_world_space')
      write_points(all_surface_points_np, 'all_surface_world_space')
      write_points(surface_camera_space_np, 'chosen_surface_camera_space')
      write_points(reprojected_camera_np,
                   'all_reprojected_depth_map_camera_space')
      write_points(flat_nic_xyz_np, 'all_depth_map_nic')
      write_points(flat_nic_xyz_np, 'chosen_depth_map_nic',
                   nonzero_depth_mask_np)
      write_points(reprojected_camera_np,
                   'valid_reprojected_depth_map_camera_spce',
                   nonzero_depth_mask_np)
      point_output_fname = os.path.join(output_dir, 'world_space_points.npy')
      np.save(point_output_fname, flat_xyz_locations_np)
      nzdm_output_fname = os.path.join(output_dir, 'mask.npy')
      np.save(nzdm_output_fname, nonzero_depth_mask_np)


if __name__ == '__main__':
  googletest.main()

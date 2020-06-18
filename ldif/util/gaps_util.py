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
"""Utilties for working with the GAPS geometric processing library."""

import math
import os
import subprocess as sp
import tempfile

import numpy as np


# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import file_util
from ldif.util import geom_util_np
from ldif.util import np_util
from ldif.util import py_util
from ldif.util import path_util
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order


def _setup_cam(camera):
  """Generates a -camera input string for GAPS viewers."""
  if camera == 'fixed':
    init_camera = ('1.0451 1.17901 0.630437 '
                   '-0.614259 -0.695319 -0.373119 '
                   '-0.547037 0.715996 -0.433705')
  elif camera == 'default':
    init_camera = None
  else:
    init_camera = camera
  if init_camera is not None:
    init_camera = ' -camera %s' % init_camera
  else:
    init_camera = ''
  return init_camera


def ptsview(pts, mesh=None, camera='fixed'):
  """Interactively visualizes a pointcloud alongside an optional mesh."""
  with py_util.py2_temporary_directory() as d:
    ptspath = _make_pts_input_str(d, pts, allow_none=False)
    init_camera = _setup_cam(camera)
    mshpath = ''
    if mesh:
      mshpath = d + '/m.ply'
      file_util.write_mesh(mshpath, mesh)
      mshpath = ' ' + mshpath
    cmd = '%s/ptsview %s%s%s' % (path_util.gaps_path(), ptspath, mshpath,
                                 init_camera)
    log.info(cmd)
    sp.check_output(cmd, shell=True)


def mshview(mesh, camera='fixed'):
  """Interactively views a mesh."""
  with py_util.py2_temporary_directory() as d:
    init_camera = _setup_cam(camera)
    if not isinstance(mesh, list):
      mesh = [mesh]
    assert len(mesh) <= 4
    mshpath = ''
    for m, c in zip(mesh, ['m', 'n', 'o', 'p']):
      lpath = f'{d}/{c}.ply'
      mshpath += f' {lpath}'
      file_util.write_mesh(lpath, m)
    cmd = '%s/mshview %s%s' % (path_util.gaps_path(), mshpath, init_camera)
    log.info(cmd)
    sp.check_output(cmd, shell=True)


def grdview(volume, world2grid=None):
  with py_util.py2_temporary_directory() as d:
    gpath = d + '/g.grd'
    # Dummy world2grid:
    file_util.write_grd(gpath, volume, world2grid=world2grid)
    cmd = '%s/grdview %s' % (path_util.gaps_path(), gpath)
    sp.check_output(cmd, shell=True)


def gapsview(keep=True, **kwargs):
  if keep:
    d = tempfile.mkdtemp()
    _gapsview(d, **kwargs)
  else:
    with py_util.py2_temporary_directory() as d:
      _gapsview(d, **kwargs)


def _make_pts_input_str(d, pts, allow_none=True):
  """Generates a string input specifying the input pointcloud paths."""
  if pts is None:
    if not allow_none:
      raise ValueError('Input points are required for this operation.')
    return ''
  if isinstance(pts, list):
    names = ['p', 'q', 'r', 's', 't']
    path = ''
    for i, p in enumerate(pts):
      cpath = d + '/' + names[i]
      path += ' ' + file_util.write_points(cpath, p)
    return path
  else:
    return ' ' + file_util.write_points(d + '/p', pts)


def _gapsview(d,
              msh=None,
              pts=None,
              grd=None,
              world2grid=None,
              grid_threshold=0.0,
              camera='default'):
  """Interactively views a mesh, pointcloud, and/or grid at the same time."""
  assert msh is not None or pts is not None or grd is not None
  mpath = ''
  ppath = ''
  gpath = ''
  init_camera = _setup_cam(camera)
  if msh is not None:
    mpath = d + '/m.ply'
    file_util.write_mesh(mpath, msh)
    mpath = ' ' + mpath
    log.info('Mpath: %s' % mpath)
  ppath = _make_pts_input_str(d, pts)
  if grd is not None:
    gpath = d + '/g.grd'
    file_util.write_grd(gpath, grd, world2grid=world2grid)
    gpath = ' ' + gpath + ' -grid_threshold %0.6f' % grid_threshold
  cmd = '%s/gapsview%s%s%s%s' % (path_util.gaps_path(), mpath, ppath, gpath,
                                 init_camera)
  log.info(cmd)
  sp.check_output(cmd, shell=True)


def read_pts_file(path):
  """Reads a .pts or a .sdf point samples file."""
  _, ext = os.path.splitext(path)
  assert ext in ['.sdf', '.pts']
  l = 4 if ext == '.sdf' else 6
  with file_util.open_file(path, 'rb') as f:
    points = np.fromfile(f, dtype=np.float32)
  points = np.reshape(points, [-1, l])
  return points


def read_cam_file(path, verbose=False):
  """Reads a GAPS .cam file to 4x4 matrices.

  Args:
    path: filepath to a gaps .cam file with K cameras.
    verbose: Boolean. Whether to print detailed file info.

  Returns:
    cam2world: Numpy array with shape [K, 4, 4].
    xfov: Numpy array with shape [K]. The x field-of-view in GAPS' format (which
      is the half-angle in radians).
  """
  lines = file_util.readlines(path)
  if verbose:
    log.info('There are %i cameras in %s.' % (len(lines), path))
  cam2worlds, xfovs = [], []
  for i, l in enumerate(lines):
    vals = [float(x) for x in l.split(' ') if x]
    if len(vals) != 12:
      raise ValueError(
          'Failed reading %s: Expected 12 items on line %i, but read %i.' %
          (path, i, len(vals)))
    viewpoint = np.array(vals[0:3]).astype(np.float32)
    towards = np.array(vals[3:6]).astype(np.float32)
    up = np.array(vals[6:9]).astype(np.float32)
    right = np.cross(towards, up)
    right = right / np.linalg.norm(right)
    xfov = vals[9]
    # 11th is yfov but GAPS ignores it and recomputes.
    # 12th is a 'score' that is irrelevant.
    towards = towards / np.linalg.norm(towards)
    up = np.cross(right, towards)
    up = up / np.linalg.norm(up)
    # aspect = float(height) / float(width)
    # yf = math.atan(aspect * math.tan(xfov))
    rotation = np.stack([right, up, -towards], axis=1)
    rotation_4x4 = np.eye(4)
    rotation_4x4[:3, :3] = rotation
    cam2world = rotation_4x4.copy()
    cam2world[:3, 3] = viewpoint
    cam2worlds.append(cam2world)
    xfovs.append(xfov)
  cam2worlds = np.stack(cam2worlds, axis=0).astype(np.float32)
  xfovs = np.array(xfovs, dtype=np.float32)
  return cam2worlds, xfovs


def batch_gaps_depth_image_to_cam_image(depth_image, xfov=None):
  """Converts a batch of GAPS depth images to camera-space images.

  Args:
    depth_image: Numpy array with shape [batch_size, height, width] or
      [batch_size, height, width, 1].
    xfov: The xfov of each image in the GAPS format (half-angle in radians).
      Either a numpy array with shape [batch_size] or [1] or a float. If it is
      None, it defaults to 0.5 for all batch items (the GAPS default).

  Returns:
    Numpy array with shape [batch_size, height, width, 3].
  """
  batch_size = depth_image.shape[0]
  assert len(depth_image.shape) in [3, 4]
  if xfov is None:
    xfov = 0.5
  if isinstance(xfov, float):
    xfov = np.ones([batch_size], dtype=np.float32) * xfov
  else:
    assert len(xfov.shape) in [0, 1]
    if not xfov.shape:
      xfov = np.reshape(xfov, [1])
    if xfov.shape[0] == 1:
      xfov = np.tile(xfov, [batch_size])
    assert xfov.shape[0] == batch_size
  out = []
  for i in range(batch_size):
    out.append(gaps_depth_image_to_cam_image(depth_image[i, ...],
                                             xfov[i]))
  return np.stack(out)


def gaps_depth_image_to_cam_image(depth_image, xfov=0.5):
  """Converts a GAPS depth image to a camera-space image.

  Args:
    depth_image: Numpy array with shape [height, width] or [height, width, 1].
      The depth in units (not in 1000ths of units, which is what GAPS writes).
    xfov: The xfov of the image in the GAPS format (half-angle in radians).

  Returns:
    cam_image: array with shape [height, width, 3].
  """
  height, width = depth_image.shape[0:2]
  depth_image = np.reshape(depth_image, [height, width])
  pixel_coords = np_util.make_coordinate_grid(
      height, width, is_screen_space=False, is_homogeneous=False)
  nic_x = 2 * pixel_coords[:, :, 0] - 1.0
  nic_y = 2 * pixel_coords[:, :, 1] - 1.0
  nic_d = -depth_image
  aspect = height / float(width)
  yfov = math.atan(aspect * math.tan(xfov))
  intrinsics_00 = 1.0 / math.tan(xfov)
  intrinsics_11 = 1.0 / math.tan(yfov)
  cam_x = nic_x * -nic_d / intrinsics_00
  cam_y = nic_y * nic_d / intrinsics_11
  cam_z = nic_d
  return np.stack([cam_x, cam_y, cam_z], axis=2)


def read_depth_im(path):
  """Loads a GAPS depth image stored as a 16-bit monochromatic PNG."""
  return file_util.read_image(path) / 1000.0


def read_normals_im(px, py, pz):
  """Loads a GAPS normal image stored as individual nx, ny, nz PNGs."""
  ns = [file_util.read_image(p) for p in [px, py, pz]]
  ns = np.stack(ns, axis=-1)
  norm = np.linalg.norm(ns, axis=-1, keepdims=True)
  ns /= np.maximum(norm, 1e-8)
  return ns


def read_normals_dir(directory, image_count):
  """Loads a directory of GAPS normal images storage as nx/ny/nz PNGs."""
  images = []
  for i in range(image_count):
    base = f'{directory}/{str(i).zfill(6)}_depth'
    image = read_normals_im(f'{base}_nx.png', f'{base}_ny.png',
                            f'{base}_nz.png')
    images.append(image)
  return np.stack(images)


def depth_path_name(depth_dir, idx):
  """Generates the GAPS filename for a depth image from its index and dir."""
  return os.path.join(depth_dir, '%s_depth.png' % str(idx).zfill(6))


def read_depth_directory(depth_dir, im_count):
  """Reads the images in a directory of depth images made by scn2img.

  Args:
    depth_dir: Path to the root directory containing the scn2img output images.
    im_count: The number of images to read. Will read images with indices
      range(im_count).

  Returns:
    Numpy array with shape [im_count, height, width]. Dimensions determined from
      file.
  """
  depth_ims = []
  for i in range(im_count):
    path = depth_path_name(depth_dir, i)
    depth_ims.append(read_depth_im(path))
  depth_ims = np.stack(depth_ims)
  assert len(depth_ims.shape) == 3
  return depth_ims


def transform_r2n2_depth_image_to_gaps_frame(depth_im, idx, e):
  """Transforms a depth image predicted in the r2n2 space to the GAPS space."""
  # depth_im = self.r2n2_depth_images[idx, ...]
  # TODO(kgenova) Add support for predicted extrinsics.
  is_valid = depth_im != 0.0
  is_valid = np.reshape(is_valid, [224, 224])

  # Depth2cam:
  cam_im = gaps_depth_image_to_cam_image(depth_im, e.r2n2_xfov[idx])
  # Cam2world
  xyz_im = geom_util_np.apply_4x4(
      cam_im, e.r2n2_cam2world[idx, ...], are_points=True)
  mask = np_util.make_mask(depth_im)
  xyz_im = np_util.zero_by_mask(mask, xyz_im).astype(np.float32)
  return xyz_im

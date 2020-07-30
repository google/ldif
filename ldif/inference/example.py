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
"""A single training example."""

import importlib
import math
import os
import time

import numpy as np
import pandas as pd
from scipy.ndimage import interpolation

# ldif is an internal package, and should be imported last.
# pylint: disable=g-bad-import-order
from ldif.inference import util as inference_util
from ldif.util import file_util
from ldif.util import gaps_util
from ldif.util import geom_util
from ldif.util import geom_util_np
from ldif.util import np_util
from ldif.util import path_util
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order

importlib.reload(gaps_util)


synset_to_cat = {
    '02691156': 'airplane',
    '02933112': 'cabinet',
    '03001627': 'chair',
    '03636649': 'lamp',
    '04090263': 'rifle',
    '04379243': 'table',
    '04530566': 'watercraft',
    '02828884': 'bench',
    '02958343': 'car',
    '03211117': 'display',
    '03691459': 'speaker',
    '04256520': 'sofa',
    '04401088': 'telephone',
    '2472293': 'bodyshapes',
    '2472294': 'caesar',
    '2472295': 'surreal',
}

cat_to_synset = {v: k for k, v in synset_to_cat.items()}


def _parse_synset_or_cat(synset_or_cat):
  """Attempts to turn wordnet synsets into readable class names."""
  if synset_or_cat in synset_to_cat:
    synset = synset_or_cat
  else:
    if synset_or_cat not in cat_to_synset:
      log.verbose(f'{synset_or_cat} is not a recognized class or synset. If you'
                  ' are not testing on shapenet-13 or a subset, this can be '
                  'safely ignored.')
      return synset_or_cat, synset_or_cat
    synset = cat_to_synset[synset_or_cat]
  cat = synset_to_cat[synset]
  return synset, cat


class ExampleGenerator(object):
  """An object that enables the random generation of dataset examples."""

  def __init__(self, is_local):
    if is_local:
      self.path = os.path.join(path_util.get_path_to_ldif_root(),
                               'data/hashlist.csv')
    else:
      raise ValueError('Remote hashlist no longer supported.')
    with file_util.open_file(self.path, 'rt') as f:
      self.df = pd.read_csv(
          f,
          dtype={
              'split': str,
              'synset': str,
              'hash': str,
              'synset_inv_freq': np.float32
          },
          usecols=['split', 'synset', 'hash', 'synset_inv_freq'])

  def random_example(self, split, synset_or_cat=None, equal_class_freq=True):
    """Returns a random InferenceExample.

    Args:
      split: String. One of 'train', 'val', 'test'.
      synset_or_cat: String. Either a synset, a category, or None. If provided,
        the random example will be sampled from that category.
      equal_class_freq: If True, then the returned object has an equal chance of
        being from each class in the dataset. If False, then a single random
        example from the dataset is return. This flag is irrelevant if
        synset_or_cat is not None.

    Returns:
      A random InferenceExample object.
    """
    df = self.df[self.df['split'].str.match(split)]
    # log.info('df split len: %i' % df.shape[0])
    if synset_or_cat:
      synset, _ = _parse_synset_or_cat(synset_or_cat)
      synset_exs = df[df['synset'].str.match(synset)]
      sampled = synset_exs.sample(n=1)
    else:
      weights = df['synset_inv_freq'] if equal_class_freq else None
      sampled = df.sample(weights=weights)
    split, synset, mesh_hash = sampled.values[0].tolist()[:3]
    return InferenceExample(split, synset, mesh_hash)


def assert_is_4x4(m):
  assert len(m.shape) == 2
  assert m.shape[0] == 4
  assert m.shape[1] == 4
  return m


def _get_world_pts_from_idx(idx,
                            depth,
                            world_xyz_im,
                            world_n_im,
                            target_point_count=10000):
  """Subsamples point and normal images together to make a pointcloud."""
  # TODO(kgenova) Consider computing validity directly from the normals.
  is_invalid = np.squeeze(depth[idx, ...] == 0.0)
  is_valid = np.logical_not(is_invalid)
  world_xyz = world_xyz_im.copy()[idx, ...]
  world_n = world_n_im.copy()[idx, ...]
  world_xyzn = np.concatenate([world_xyz, world_n], axis=-1)
  world_xyzn = world_xyzn[is_valid, :]
  world_xyzn = np.reshape(world_xyzn, [-1, 6])
  np.random.shuffle(world_xyzn)
  point_count = world_xyzn.shape[0]
  assert point_count > 0
  log.info('The number of valid samples for idx %i is: %i' % (idx, point_count))
  while point_count < target_point_count:
    world_xyzn = np.tile(world_xyzn, [2, 1])
    point_count = world_xyzn.shape[0]
  return world_xyzn[:target_point_count, :].astype(np.float32)


class InferenceExample(object):
  """A dataset example for inference-time use."""

  def __init__(self, split, synset_or_cat, mesh_hash=None, dynamic=False,
               verbose=True):
    self.split = split
    self.synset, self.cat = _parse_synset_or_cat(synset_or_cat)
    self.mesh_hash = mesh_hash
    self._rgb_path = None
    self._rgb_image = None
    self.__archive = None
    self._uniform_samples = None
    self._near_surface_samples = None
    self._grid = None
    self._world2grid = None
    self._gt_path = None
    self._tx = None
    self._gaps_to_occnet = None
    self._gt_mesh = None
    self._tx_path = None
    self._surface_samples = None
    self._normalized_gt_mesh = None
    self._r2n2_images = None
    self.depth_native_res = 224

    self.is_from_directory = False

    if dynamic:
      if verbose:
        log.verbose(
            'Using dynamic files, not checking ahead for file existence.')
    elif not file_util.exists(self.npz_path):
      raise ValueError('Expected a .npz at %s.' % self.npz_path)
    else:
      log.info(self.npz_path)

  @classmethod
  def from_relpath(cls, relpath):
    split, synset, mesh_hash = relpath.split('/')
    return cls(split, synset, mesh_hash)

  @classmethod
  def from_local_dataset_tokens(cls, model_directory, split, class_name,
                                mesh_name):
    return cls.from_directory(os.path.join(model_directory, split, class_name,
                                           mesh_name))

  @classmethod
  def from_directory(cls, dirpath, verbose=True):
    """Creates an example from a meshes2dataset mesh subdirectory."""
    if dirpath[-1] == '/':
      dirpath = dirpath[:-1]
    mesh_hash = os.path.splitext(os.path.basename(dirpath))[0]
    prepath = dirpath[:dirpath.rfind(mesh_hash)]
    assert prepath[-1] == '/'
    prepath = prepath[:-1]
    split, synset = prepath.split('/')[-2:]
    ex = cls(split=split, synset_or_cat=synset,
             mesh_hash=mesh_hash, dynamic=True, verbose=verbose)
    # pylint: disable=protected-access
    ex._tx_path = f'{dirpath}/orig_to_gaps.txt'
    ex._dodeca_depth_and_normal_path = f'{dirpath}/depth_and_normals.npz'
    ex._gt_path = f'{dirpath}/mesh_orig.ply'
    ex._directory_root = dirpath
    ex._grid_path = f'{dirpath}/coarse_grid.grd'
    # pylint: enable=protected-access
    ex.precomputed_surface_samples_from_dodeca_path = (
        f'{dirpath}/surface_samples_from_dodeca.pts'
    )
    ex.is_from_directory = True
    return ex

  @classmethod
  def from_npz_path(cls, npz_path):
    split, synset, hash_and_ext = npz_path.split('/')[-3:]
    mesh_hash = hash_and_ext.split('.')[0]
    inst = cls(split, synset, mesh_hash)
    if inst.npz_path != npz_path:
      raise ValueError(
          ('Internal error: provided npz path is a mismatch'
           ' with generated one: %s vs %s') % (inst.npz_path, npz_path))
    return inst

  @classmethod
  def from_elt(cls, elt):
    split, synset, mesh_hash = elt['mesh_identifier'].split('/')
    inst = cls(split, synset, mesh_hash)
    if inst.mesh_hash != mesh_hash:
      raise ValueError('Internal error: failed to parse input mesh_identifier.')
    return inst

  @property
  def npz_path(self):
    return '/DATA_PATH/shapenet-occnet/%s/%s/%s.npz' % (self.split, self.synset,
                                                        self.mesh_hash)

  @property
  def surface_samples_path(self):
    sp = self.npz_path.replace('/shapenet-occnet/',
                               '/shapenet-occnet/surface-points/').replace(
                                   '.npz', '/surface_points.npy')
    return sp

  @property
  def grid_path(self):
    if not hasattr(self, '_grid_path'):
      self._grid_path = self.surface_samples_path.replace(
          '/surface-points/', '/lowres-grids/').replace('/surface_points.npy',
                                                        '/lowres_grid.grd')
    return self._grid_path

  @property
  def dodeca_depth_and_normal_path(self):
    if not hasattr(self, '_dodeca_depth_and_normal_path'):
      self._dodeca_depth_and_normal_path = (
          f'/DATA_PATH'
          f'/{self.split}/{self.synset}/{self.mesh_hash[:-1]}.npz')
    return self._dodeca_depth_and_normal_path

  @property
  def r2n2_depth_dir(self):
    """The directory containing the GT r2n2 depth images."""
    root = '/DATA_PATH/3dr2n2-renders/3dr2n2/'
    if self.depth_native_res == 137:
      depth_dir = root + 'depth-137/%s/%s' % (self.synset, self.mesh_hash)
    elif self.depth_native_res == 224:
      depth_dir = root + 'depth-normals-npy-224/%s/%s' % (self.synset,
                                                          self.mesh_hash)
    else:
      assert False
    return depth_dir

  @property
  def r2n2_cam2v1_path(self):
    root = '/DATA_PATH/3dr2n2-renders/3dr2n2/'
    cam_path = root + 'cams/%s/%s.cam' % (self.synset, self.mesh_hash)
    return cam_path

  @property
  def v12occnet_path(self):
    root = '/DATA_PATH/v1shapenet2occnet/'
    path = root + '%s/%s.npy' % (self.synset, self.mesh_hash)
    return path

  @classmethod
  def from_rgb_path_and_split(cls, rgb_path, split):
    npz_path = inference_util.rgb_path_to_npz_path(rgb_path, split)
    inst = cls.from_npz_path(npz_path)
    inst.set_rgb_path(rgb_path)
    return inst

  def set_rgb_path(self, rgb_path):
    self._rgb_path = rgb_path

  def set_rgb_path_from_idx(self, idx):
    del idx  # Unused
    raise ValueError('Unimplemented')

  @property
  def _archive(self):
    """The backing npz file contents."""
    if self.__archive is None:
      t = time.time()
      try:
        self.__archive = file_util.read_npz(self.npz_path)
      except:
        raise ValueError('Failed to read %s' % self.npz_path)
      log('Time loading input archive: %0.2f' % (time.time() - t))
    return self.__archive

  @property
  def rgb_path(self):
    if self._rgb_path is None:
      raise ValueError('rgb_path has not been set for this example.')
    return self._rgb_path

  def set_depth_res(self, res):
    assert res in [137, 224]
    self.depth_native_res = res

  @property
  def rgb_image(self):
    if self._rgb_image is None:
      self._rgb_image = inference_util.read_png_to_float_npy_with_reraising(
          self.rgb_path)
      self._rgb_image = np.reshape(self._rgb_image, [1, 137, 137, 4])[..., :3]
    return self._rgb_image

  @property
  def r2n2_images(self):
    if self._r2n2_images is None:
      self._r2n2_images = np.reshape(self._archive['rgb_3dr2n2'],
                                     [24, 137, 137, 4])
    # log.info(list(self._archive.keys()))
    return self._r2n2_images

  def load_depth_and_normal_npz(self, path):
    depth_normal_arr = file_util.read_npz(path)['arr_0']
    depth = depth_normal_arr[..., 0]  # For backwards compat.
    normals = depth_normal_arr[..., 1:]
    return depth, normals

  @property
  def r2n2_depth_images(self):
    if not hasattr(self, '_r2n2_depth_images'):
      if self.depth_native_res == 137:
        self._r2n2_depth_images = gaps_util.read_depth_directory(
            self.r2n2_depth_dir, 24)
      elif self.depth_native_res == 224:
        self._r2n2_depth_images, self._r2n2_normal_world_images = (
            self.load_depth_and_normal_npz(self.r2n2_depth_dir + '.npz'))
    return self._r2n2_depth_images

  @property
  def r2n2_normal_world_images(self):
    if not hasattr(self, '_r2n2_normal_world_images'):
      assert self.depth_native_res == 224
      self._r2n2_depth_images, self._r2n2_normal_world_images = (
          self.load_depth_and_normal_npz(self.r2n2_depth_dir + '.npz'))
    return self._r2n2_normal_world_images

  @property
  def r2n2_normal_cam_images(self):
    """The from-depth GAPS-predicted R2N2 normal images in camera space."""
    if not hasattr(self, '_r2n2_normal_cam_images'):
      nrm_world = self.r2n2_normal_world_images
      cam2world = self.r2n2_cam2world.copy()
      cam_images = []
      for i in range(24):
        # Use the inverse-transpose of the needed matrix:
        im_i = geom_util_np.apply_4x4(
            nrm_world[i, ...], cam2world[i, :, :].T, are_points=False)
        nrm = np.linalg.norm(im_i, axis=-1, keepdims=True) + 1e-10
        im_i /= nrm
        mask = np_util.make_mask(self.r2n2_depth_images[i, ...])
        cam_images.append(np_util.zero_by_mask(mask, im_i).astype(np.float32))
      self._r2n2_normal_cam_images = np.stack(cam_images)
    return self._r2n2_normal_cam_images

  @property
  def r2n2_cam_images(self):
    if not hasattr(self, '_r2n2_cam_images'):
      cam_images = []
      for i in range(24):
        im_i = gaps_util.gaps_depth_image_to_cam_image(
            self.r2n2_depth_images[i, ...], self.r2n2_xfov[i])
        cam_images.append(im_i)
      self._r2n2_cam_images = np.stack(cam_images)
    return self._r2n2_cam_images

  @property
  def r2n2_xyz_images(self):
    """The GT R2N2 XYZ images in world space."""
    if not hasattr(self, '_r2n2_xyz_images'):
      xyz_images = []
      for i in range(24):
        im_i = geom_util_np.apply_4x4(
            self.r2n2_cam_images[i, ...],
            self.r2n2_cam2world[i, ...],
            are_points=True)
        mask = np_util.make_mask(self.r2n2_depth_images[i, ...])
        xyz_images.append(np_util.zero_by_mask(mask, im_i).astype(np.float32))
      self._r2n2_xyz_images = np.stack(xyz_images)
    return self._r2n2_xyz_images

  def set_r2n2_cam2v1(self, cam2v1):
    assert not hasattr(self, '_r2n2_cam2v1')
    self._r2n2_cam2v1 = np.reshape(cam2v1, [24, 4, 4])
    self._r2n2_xfov = 0.422204 + np.zeros([24], dtype=np.float32)

  @property
  def r2n2_cam2v1(self):
    if not hasattr(self, '_r2n2_cam2v1'):
      self._r2n2_cam2v1, self._r2n2_xfov = gaps_util.read_cam_file(
          self.r2n2_cam2v1_path)
    return self._r2n2_cam2v1

  @property
  def r2n2_xfov(self):
    if not hasattr(self, '_r2n2_xfov'):
      self._r2n2_cam2v1, self._r2n2_xfov = gaps_util.read_cam_file(
          self.r2n2_cam2v1_path)
    return self._r2n2_xfov

  @property
  def r2n2_viewpoints(self):
    """A view from the v1 frame offset to the 24 r2n2 cameras. Shape [24, 3]."""
    if not hasattr(self, '_r2n2_viewpoints'):

      def compute_pose(metadata_line):
        """Computes the spherical coordinate 3-DOF pose parameters."""
        # TODO(kgenova) Implement
        elements = metadata_line.split(' ')
        assert len(elements) == 5
        fake_azi = float(elements[0])
        fake_ele = float(elements[1])
        fake_dist = float(elements[3])
        fake_azi = -fake_azi * math.pi / 180.0
        fake_ele = math.pi / 2.0 - fake_ele * math.pi / 180.0
        fake_dist = fake_dist * 1.75
        x = fake_dist * math.sin(fake_ele) * math.cos(fake_azi)
        y = fake_dist * math.cos(fake_ele)
        z = -fake_dist * math.sin(fake_ele) * math.sin(fake_azi)
        return np.array([x, y, z], dtype=np.float32)

      metadata_path = (
          f'/DATA_PATH/3dr2n2-renders/'
          f'{self.synset}/{self.mesh_hash}/rendering/rendering_metadata.txt')
      lines = file_util.read_lines(metadata_path)
      self._r2n2_viewpoints = np.stack([compute_pose(line) for line in lines])
    return self._r2n2_viewpoints

  @property
  def v1_watertight_gt_mesh(self):
    if not hasattr(self, '_v1_watertight_gt_mesh'):
      m = self.gt_mesh.copy()
      occnet2v1 = self.occnet2v1
      m.apply_transform(occnet2v1)
      self._v1_watertight_gt_mesh = m
    return self._v1_watertight_gt_mesh

  @property
  def v12occnet(self):
    if not hasattr(self, '_v12occnet'):
      self._v12occnet = assert_is_4x4(file_util.read_np(self.v12occnet_path))
    return self._v12occnet

  @property
  def occnet2v1(self):
    if not hasattr(self, '_occnet2v1'):
      self._occnet2v1 = assert_is_4x4(np.linalg.inv(self.v12occnet))
    return self._occnet2v1

  @property
  def gaps2occnet(self):
    if not hasattr(self, '_gaps2occnet'):
      self._gaps2occnet = np.linalg.inv(self.occnet2gaps)
    return self._gaps2occnet

  @property
  def gaps2v1(self):
    if not hasattr(self, '_gaps2v1'):
      self._gaps2v1 = np.reshape(
          np.matmul(self.occnet2v1, self.gaps2occnet),
          [4, 4]).astype(np.float32)
    return self._gaps2v1

  @property
  def depr_depth_images(self):
    """A stack of 20 depth images of the shape."""
    if not hasattr(self, '_depr_depth_images'):
      depth_images = self._archive['gaps_depth']
      self._depr_depth_images = depth_images.reshape([1, 20, 224, 224, 1])
    return self._depr_depth_images

  @property
  def depth_images(self):
    if not hasattr(self, '_depth_images'):
      self._depth_images, self._cam_normal_images = self.load_depth_and_normal_npz(
          self.dodeca_depth_and_normal_path)
      self._depth_images = np.reshape(self._depth_images, [1, 20, 224, 224, 1])
      self._depth_images *= 1000.0
    return self._depth_images

  @property
  def cam_normal_images(self):
    """Normal images computed directly from the depth dodeca images."""
    if not hasattr(self, '_cam_normal_images'):
      self._depth_images, self._cam_normal_images = self.load_depth_and_normal_npz(
          self.dodeca_depth_and_normal_path)
      self._depth_images = np.reshape(self._depth_images, [1, 20, 224, 224, 1])
      # Report in 1000-ths for backwards compatibility:
      self._depth_images *= 1000.0
    return self._cam_normal_images

  @property
  def world_normal_images(self):
    """Normal images transformed to world space."""
    if not hasattr(self, '_world_normal_images'):
      cam_normals = self.cam_normal_images.copy()
      cam2world = self.dodeca_cam2world.copy()
      world_normals = []
      for i in range(20):
        im_i = cam_normals[i, ...]
        # Normals are transformed by the inverse transpose
        cam2world_invt = np.linalg.inv(cam2world[i, ...]).T
        im_i = geom_util_np.apply_4x4(im_i, cam2world_invt, are_points=False)
        nrm = np.linalg.norm(im_i, axis=-1, keepdims=True) + 1e-10
        im_i /= nrm
        mask = np_util.make_mask(self.depth_images[0, i, ...])
        world_normals.append(
            np_util.zero_by_mask(mask, im_i).astype(np.float32))
      self._world_normal_images = np.stack(world_normals)
    return self._world_normal_images

  @property
  def surface_samples(self):
    """Point samples with normals from the shape's surface."""
    if self._surface_samples is None:
      self._surface_samples = self.get_surface_samples(10000)
    return self._surface_samples

  @property
  def dodeca_cam2world(self):
    if not hasattr(self, '_dodeca_cam2world'):
      self._dodeca_cam2world = geom_util.get_dodeca_camera_to_worlds()
    return self._dodeca_cam2world

  @property
  def cam_xyz_images_from_dodeca(self):
    if not hasattr(self, '_cam_xyz_images_from_dodeca'):
      depth_ims = self.depth_images.copy() / 1000.0
      cam_images = []
      for i in range(20):
        im = depth_ims[0, i, ...]
        cam_images.append(gaps_util.gaps_depth_image_to_cam_image(im, xfov=0.5))
      self._cam_xyz_images_from_dodeca = np.stack(cam_images)
    return self._cam_xyz_images_from_dodeca

  @property
  def world_xyz_images_from_dodeca(self):
    """The world-space XYZ image derived from the 20 Dodeca images."""
    if not hasattr(self, '_world_xyz_images_from_dodeca'):
      world_images = []
      for i in range(20):
        im_i = geom_util_np.apply_4x4(
            self.cam_xyz_images_from_dodeca[i, ...],
            self.dodeca_cam2world[i, ...],
            are_points=True)
        mask = np_util.make_mask(self.depth_images[0, i, ...])
        world_images.append(np_util.zero_by_mask(mask, im_i).astype(np.float32))
      self._world_xyz_images_from_dodeca = np.stack(world_images)
    return self._world_xyz_images_from_dodeca

  def get_surface_samples_from_single_dodeca_image(self, idx):
    """Computes the XYZ+N samples from one dodeca image, tiling as necessary."""
    depth_im = self.depth_images[0, idx, ...]
    is_valid = depth_im != 0.0
    is_valid = np.reshape(is_valid, [224, 224])
    world_xyz = self.world_xyz_images_from_dodeca.copy()[idx, ...]
    world_n = self.world_normal_images.copy()[idx, ...]
    world_xyzn = np.concatenate([world_xyz, world_n], axis=-1)
    world_xyzn = world_xyzn[is_valid, :]
    world_xyzn = np.reshape(world_xyzn, [-1, 6])
    np.random.shuffle(world_xyzn)
    point_count = world_xyzn.shape[0]
    assert point_count > 0
    log.info('The number of valid samples for idx %i is: %i' %
             (idx, point_count))
    while point_count < 10000:
      world_xyzn = np.tile(world_xyzn, [2, 1])
      point_count = world_xyzn.shape[0]
    return world_xyzn[:10000, :]

  def get_surface_samples_from_single_r2n2_depth_image(self, idx):
    """Computes surface samples from the idx-th depth_image."""
    depth_im = self.r2n2_depth_images[idx, ...]
    is_valid = depth_im != 0.0
    is_valid = np.reshape(is_valid, [224, 224])
    world_xyz = self.r2n2_xyz_images.copy()[idx, ...]
    world_n = self.r2n2_normal_world_images.copy()[idx, ...]
    world_xyzn = np.concatenate([world_xyz, world_n], axis=-1)
    world_xyzn = world_xyzn[is_valid, :]
    world_xyzn = np.reshape(world_xyzn, [-1, 6])
    np.random.shuffle(world_xyzn)
    point_count = world_xyzn.shape[0]
    assert point_count > 0
    log.info('The number of valid samples for idx %i is: %i' %
             (idx, point_count))
    while point_count < 10000:
      world_xyzn = np.tile(world_xyzn, [2, 1])
      point_count = world_xyzn.shape[0]
    return world_xyzn[:10000, :]

  @property
  def surface_sample_count(self):
    if not hasattr(self, '_surface_sample_count'):
      self._surface_sample_count = 10000
    return self._surface_sample_count

  @surface_sample_count.setter
  def surface_sample_count(self, count):
    assert isinstance(count, int)
    assert count >= 1
    assert count <= 100000
    self._surface_sample_count = count


  @property
  def precomputed_surface_samples_from_dodeca(self):
    if not hasattr(self, '_precomputed_surface_samples_from_dodeca'):
      if not self.is_from_directory:
        raise ValueError('Precomputed surface samples are only'
                         ' available with a from_directory example.')
      if not os.path.isfile(self.precomputed_surface_samples_from_dodeca_path):
        raise ValueError('Dodeca surface samples have not been precomputed at '
                         f'{self.precomputed_surface_samples_from_dodeca_path}')
      full_samples = gaps_util.read_pts_file(
          self.precomputed_surface_samples_from_dodeca_path)
      orig_count = 100000
      assert full_samples.shape[0] == orig_count
      assert full_samples.shape[1] == 6
      assert full_samples.dtype == np.float32
      assert full_samples.shape[0] > 1
      while full_samples.shape[0] < self.surface_sample_count:
        log.verbose(f'Doubling samples from {full_samples.shape[0]} to'
                    f' {2*full_samples.shape[0]}')
        full_samples = np.tile(full_samples, [2, 1])
      self._precomputed_surface_samples_from_dodeca = (
          full_samples[np.random.choice(orig_count, self.surface_sample_count, replace=False), :])
    return self._precomputed_surface_samples_from_dodeca


  @property
  def surface_samples_from_dodeca(self):
    """10K surface point samples with normals computed from the dodecahedron."""
    if not hasattr(self, '_surface_samples_from_dodeca'):
      depth_ims = self.depth_images.copy() / 1000.0
      is_valid = depth_ims != 0.0
      is_valid = np.reshape(is_valid, [20, 224, 224])
      world_xyz = self.world_xyz_images_from_dodeca.copy()
      world_n = self.world_normal_images.copy()
      world_xyzn = np.concatenate([world_xyz, world_n], axis=-1)
      world_xyzn = world_xyzn[is_valid, :]
      world_xyzn = np.reshape(world_xyzn, [-1, 6])
      np.random.shuffle(world_xyzn)
      assert world_xyzn.shape[0] > 1
      while world_xyzn.shape[0] < self.surface_sample_count:
        log.verbose(f'Tiling samples from {world_xyzn.shape[0]} to'
                    f' {2*world_xyzn.shape[0]}')
        world_xyzn = np.tile(world_xyzn, [2, 1])
      self._surface_samples_from_dodeca = world_xyzn[:self
                                                     .surface_sample_count, :]
    return self._surface_samples_from_dodeca

  def get_surface_samples(self, sample_count):
    pts = file_util.read_np(self.surface_samples_path)
    np.random.shuffle(pts)
    pts = pts[:sample_count, :]
    return pts

  @property
  def near_surface_samples(self):
    """The xyz-sdf samples that are biased to be near the GT surface."""
    if self._near_surface_samples is None:
      if self.is_from_directory:
        nss_sample_path = f'{self._directory_root}/nss_points.sdf'
        nss = gaps_util.read_pts_file(nss_sample_path)
        # log.info(f'The points have shape {nss.shape}')
      else:
        nss = self._archive['axis_samples']
      self._near_surface_samples = np.reshape(nss,
                                              [100000, 4]).astype(np.float32)
    return self._near_surface_samples

  @property
  def uniform_samples(self):
    """1000 XYZ-sign samples generated uniformly in the volume."""
    if self._uniform_samples is None:
      if self.is_from_directory:
        uniform_sample_path = f'{self._directory_root}/uniform_points.sdf'
        uniform_samples = gaps_util.read_pts_file(uniform_sample_path)
        # log.info(f'The uniform points have shape {uniform_samples.shape}')
      else:
        uniform_samples = self._archive['uniform_samples']
      self._uniform_samples = np.reshape(uniform_samples,
                                         [100000, 4]).astype(np.float32)
    return self._uniform_samples

  @property
  def mesh_name(self):
    return '%s|%s' % (self.synset, self.mesh_hash)

  @property
  def grid(self):
    if self._grid is None:
      self._world2grid, self._grid = file_util.read_grd(self.grid_path)
    return self._grid

  @property
  def world2grid(self):
    if self._world2grid is None:
      self._world2grid, self._grid = file_util.read_grd(self.grid_path)
    return self._world2grid

  @property
  def gt_path(self):
    if self._gt_path is None:
      self._gt_path = ('/DATA_PATH/occnet-plys/'
                       '%s/%s/%s/model_occnet.ply') % (self.split, self.synset,
                                                       self.mesh_hash)
    return self._gt_path

  @property
  def normalized_gt_mesh(self):
    if self._normalized_gt_mesh is None:
      m = self.gt_mesh.copy()
      occnet2gaps = self.occnet2gaps
      m.apply_transform(occnet2gaps)
      self._normalized_gt_mesh = m
    return self._normalized_gt_mesh

  @property
  def v1_gt_mesh(self):
    if not hasattr(self, '_v1_gt_mesh'):
      m = self.gt_mesh.copy()
      m.apply_transform(self.occnet2v1)
      self._v1_gt_mesh = m
    return self._v1_gt_mesh

  @property
  def tx_path(self):
    if not self._tx_path:
      self._tx_path = ('/DATA_PATH/occnet-to-gaps/'
                       '%s/%s/%s/occnet_to_gaps.txt') % (
                           self.split, self.synset, self.mesh_hash)
    return self._tx_path

  @property
  def r2n2_cam2world(self):
    """The camera space -> world space transformations for 3D-R2N2's images."""
    if not hasattr(self, '_r2n2_cam2world'):
      ms = []
      for i in range(24):
        cam2v1 = assert_is_4x4(self.r2n2_cam2v1[i, ...])
        v12occnet = assert_is_4x4(self.v12occnet)
        occnet2gaps = assert_is_4x4(self.occnet2gaps)
        cam2occnet = np.matmul(v12occnet, cam2v1)
        cam2gaps = np.matmul(occnet2gaps, cam2occnet)
        ms.append(assert_is_4x4(cam2gaps))
      self._r2n2_cam2world = np.stack(ms).astype(np.float32)
    return self._r2n2_cam2world

  @property
  def occnet2gaps(self):
    return self.tx

  @property
  def tx(self):
    if self._tx is None:
      tx = file_util.read_txt_to_np(self.tx_path)
      self._tx = np.reshape(tx, [4, 4])
    return self._tx

  @property
  def gaps_to_occnet(self):
    if self._gaps_to_occnet is None:
      self._gaps_to_occnet = np.linalg.inv(self.tx)
    return self._gaps_to_occnet

  @property
  def gt_mesh(self):
    if self._gt_mesh is None:
      self._gt_mesh = file_util.read_mesh(self.gt_path)
    return self._gt_mesh

  @property
  def gt_mesh_str(self):
    if not hasattr(self, '_gt_mesh_str'):
      self._gt_mesh_str = file_util.readbin(self.gt_path)
    return self._gt_mesh_str

  @property
  def max_depth_normals_path(self):
    return ('/DATA_PATH/max/'
            'depth-normals-npz/%s/%s/%s.npz') % (self.split, self.synset,
                                                 self.mesh_hash)

  def _load_max_depth_normals(self):
    path = self.max_depth_normals_path
    dn = file_util.read_npz(path)['arr_0']
    depth = dn[:, :, :, 0:1]
    normals = dn[:, :, :, 1:]
    return depth, normals

  @property
  def max_depth_512(self):
    if not hasattr(self, '_max_depth_512'):
      self._max_depth_512, self._max_pred_cam_normals_512 = (
          self._load_max_depth_normals())
    return self._max_depth_512

  @property
  def max_pred_cam_normals_512(self):
    if not hasattr(self, '_max_pred_normals_512'):
      self._max_depth_512, self._max_pred_cam_normals_512 = (
          self._load_max_depth_normals())
    return self._max_pred_cam_normals_512

  @property
  def max_cam_xyz(self):
    if not hasattr(self, '_max_cam_xyz'):
      depths = self.max_depth_512
      self._max_cam_xyz = geom_util_np.depth_images_to_cam_images(depths)
    return self._max_cam_xyz

  @property
  def bts_cam_xyz(self):
    if not hasattr(self, '_bts_cam_xyz'):
      depths = self.bts_depth_480
      self._bts_cam_xyz = gaps_util.batch_gaps_depth_image_to_cam_image(
          depths, self.r2n2_xfov)
    return self._bts_cam_xyz

  @property
  def bts_cam2world(self):
    if not hasattr(self, '_bts_cam2world'):
      self._bts_cam2world = self.r2n2_cam2world.copy()
    return self._bts_cam2world

  @property
  def max_cam2occnet_path(self):
    return ('/DATA_PATH/'
            'occnet_scan_64_pkls_noclip/%s/%s/%s/occnet_scan.pkl') % (
                self.split, self.synset, self.mesh_hash)

  @property
  def max_cam2occnet(self):
    if not hasattr(self, '_max_cam2occnet'):
      self._max_cam2occnet = file_util.read_py2_pkl(
          self.max_cam2occnet_path)['cam_poses']
    return self._max_cam2occnet

  @property
  def max_cam2world(self):
    """The cam2world matrices for the max depth images."""
    if not hasattr(self, '_max_cam2world'):
      occnet2gaps = self.occnet2gaps
      cam2occnet = self.max_cam2occnet
      assert cam2occnet.shape[0] == 16
      assert cam2occnet.shape[1] == 4
      assert cam2occnet.shape[2] == 4
      assert occnet2gaps.shape[0] == 4
      assert occnet2gaps.shape[1] == 4
      cam2worlds = []
      for i in range(16):
        cam2worlds.append(np.matmul(occnet2gaps, cam2occnet[i, :, :]))
      self._max_cam2world = np.stack(cam2worlds)
    return self._max_cam2world

  @property
  def max_world_xyz(self):
    if not hasattr(self, '_max_world_xyz'):
      cam = self.max_cam_xyz
      world = geom_util_np.batch_apply_4x4(cam, self.max_cam2world)
      is_invalid = np.squeeze(self.max_depth_512 == 0.0)
      world[is_invalid, :] = 0.0
      self._max_world_xyz = world
    return self._max_world_xyz

  @property
  def bts_world_xyz(self):
    if not hasattr(self, '_bts_world_xyz'):
      cam = self.bts_cam_xyz
      world = geom_util_np.batch_apply_4x4(cam, self.bts_cam2world)
      is_invalid = np.squeeze(self.bts_depth_480 == 0.0)
      world[is_invalid, :] = 0.0
      self._bts_world_xyz = world
    return self._bts_world_xyz

  @property
  def max_world_normals(self):
    if not hasattr(self, '_max_world_normals'):
      self._max_world_normals = geom_util_np.transform_normals(
          self.max_pred_cam_normals_512, self.max_cam2world)
    return self._max_world_normals

  @property
  def bts_world_normals(self):
    if not hasattr(self, '_bts_world_normals'):
      self._bts_world_normals = geom_util_np.transform_normals(
          self.bts_pred_cam_normals_480, self.bts_cam2world)
    return self._bts_world_normals

  def get_bts_world_pts_from_idx(self, idx):
    return _get_world_pts_from_idx(idx, self.bts_depth_480, self.bts_world_xyz,
                                   self.bts_world_normals)

  def get_max_world_pts_from_idx(self, idx):
    return _get_world_pts_from_idx(idx, self.max_depth_512, self.max_world_xyz,
                                   self.max_world_normals)

  def _batch_resize(self, ims, res, strategy='nearest'):
    """Resizes a batch of images to a new resolution."""
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}[strategy]
    bs = ims.shape[0]
    out = []
    log.info('Input ims shape: %s' % repr(ims.shape))
    has_extra_dim = len(ims.shape) == 4
    if not has_extra_dim:
      ims = ims[..., np.newaxis]
    h, w = ims.shape[1:3]
    for i in range(bs):
      o = interpolation.zoom(
          ims[i, ...], [res[0] / h, res[1] / w, 1.0], np.float32, order=order)
      out.append(o)
    out = np.stack(out)
    if not has_extra_dim:
      out = np.reshape(out, out.shape[:-1])
    return out

  @property
  def max_world_xyz_224(self):
    if not hasattr(self, '_max_world_xyz_224'):
      self._max_world_xyz_224 = self._batch_resize(self.max_world_xyz.copy(),
                                                   (224, 224), 'nearest')
    return self._max_world_xyz_224

  @property
  def bts_world_xyz_224(self):
    if not hasattr(self, '_bts_world_xyz_224'):
      self._bts_world_xyz_224 = self._batch_resize(self.bts_world_xyz.copy(),
                                                   (224, 224), 'nearest')
    return self._bts_world_xyz_224

  @property
  def max_world_normals_224(self):
    if not hasattr(self, '_max_world_normals_224'):
      self._max_world_normals_224 = self._batch_resize(
          self.max_world_normals.copy(), (224, 224), 'nearest')
    return self._max_world_normals_224

  @property
  def bts_depth_224(self):
    if not hasattr(self, '_bts_depth_224'):
      log.info('BTS depth 480 shape: %s' % repr(self.bts_depth_480.shape))
      self._bts_depth_224 = self._batch_resize(self.bts_depth_480.copy(),
                                               (224, 224), 'nearest')
    return self._bts_depth_224

  @property
  def bts_depth_480(self):
    """Predicted depth from the 3D-R2N2 image using the BTS algorithm."""
    if not hasattr(self, '_bts_depth_480'):
      ims = []
      for i in range(24):
        use_gt = False
        if use_gt:
          path = ('/DATA_PATH/bts-pred/depth-gt/'
                  '%s/%s/%s_depth.png') % (self.synset, self.mesh_hash,
                                           str(i).zfill(6))
        else:
          fpath = 'filtered' if self.split == 'train' else 'filtered-val'
          path = (
              '/DATA_PATH/bts-pred/%s/'
              '3dr2n2-renders-hr_3dr2n2-renders-hr-%s-%s-rendering-%s_trimmed.png'
          ) % (fpath, self.synset, self.mesh_hash, str(i).zfill(2))
        ims.append(gaps_util.read_depth_im(path))
      self._bts_depth_480 = np.stack(ims)
      self._bts_depth_480[self._bts_depth_480 != 0.0] -= 115.862 / 1000.0
    return self._bts_depth_480

  @property
  def bts_pred_cam_normals_480(self):
    """Normals predicted via the BTS algorithm followed by GAPS conf2img."""
    if not hasattr(self, '_bts_normals_480'):
      ims = []
      for i in range(24):
        if self.split == 'train':
          path_base = ('/DATA_PATH/bts-pred/<n>/'
                       '3dr2n2-renders-hr_3dr2n2-renders-hr-%s-'
                       '%s-rendering-%s_trimmed_<n>.png') % (
                           self.synset, self.mesh_hash, str(i).zfill(2))
        elif self.split == 'val':
          path_base = ('/DATA_PATH/bts-pred/'
                       'normals/val/<n>/3dr2n2-renders-hr_3dr2n2-renders-hr-%s-'
                       '%s-rendering-%s_trimmed_<n>.png') % (
                           self.synset, self.mesh_hash, str(i).zfill(2))
        ims.append(
            gaps_util.read_normals_im(
                path_base.replace('<n>', 'nx'), path_base.replace('<n>', 'ny'),
                path_base.replace('<n>', 'nz')))
      self._bts_normals_480 = np.stack(ims)
    return self._bts_normals_480

  @property
  def max_depth_224(self):
    if not hasattr(self, '_max_depth_224'):
      self._max_depth_224 = self._batch_resize(self.max_depth_512.copy(),
                                               (224, 224), 'nearest')
    return self._max_depth_224

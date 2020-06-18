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
"""Utilities for working with files."""

import os
import pickle
import struct

import numpy as np
import pandas as pd
from PIL import Image

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import base_util
from ldif.util import mesh_util
# pylint: enable=g-bad-import-order

glob = base_util.FS.glob
exists = base_util.FS.exists
mkdir = base_util.FS.mkdir
makedirs = base_util.FS.makedirs
cp = base_util.FS.cp
rm = base_util.FS.rm
open_file = base_util.FS.open
log = base_util.LOG


def readlines(p):
  with base_util.FS.open(p, 'rt') as f:
    return [x for x in f.read().split('\n') if x]


def readbin(p):
  with base_util.FS.open(p, 'rb') as f:
    return f.read()


def writebin(p, s):
  with base_util.FS.open(p, 'wb') as f:
    f.write(s)


def writetxt(p, s):
  with base_util.FS.open(p, 'wt') as f:
    f.write(s)


def write_np(path, arr):
  with base_util.FS.open(path, 'wb') as f:
    np.save(f, arr)


def read_grd(path):
  """Reads a GAPS .grd file into a (tx, grd) pair."""
  with base_util.FS.open(path, 'rb') as f:
    content = f.read()
  res = struct.unpack('iii', content[:4 * 3])
  vcount = res[0] * res[1] * res[2]
  # if res[0] != 32 or res[1] != 32 or res[2] != 32:
  #   raise ValueError(f'Expected a resolution of 32^3 but got '
  #                    f'({res[0]}, {res[1]}, {res[2]}) for example {path}.')
  content = content[4 * 3:]
  tx = struct.unpack('f' * 16, content[:4 * 16])
  tx = np.array(tx).reshape([4, 4]).astype(np.float32)
  content = content[4 * 16:]
  grd = struct.unpack('f' * vcount, content[:4 * vcount])
  grd = np.array(grd).reshape(res).astype(np.float32)
  return tx, grd


def read_sif_v1(path, verbose=False):
  """Reads a version 1 SIF .txt file and returns a numpy array."""
  text = readlines(path)
  header = text[0]
  # header_tokens = header.split(' ')
  if header != 'SIF':
    raise ValueError(f'Path {path} does not contain a SIF file.')
  shape_count, version, implicit_len = [int(x) for x in text[1].split(' ')]
  if version != 0:
    raise ValueError(f'Expected SIF version identifier 0 but got {version}.')
  assert shape_count > 0
  assert implicit_len > 0
  assert len(text) == shape_count + 2
  rep = []
  for idx in range(shape_count):
    row = text[idx + 2]
    elements = row.split(' ')
    explicits = [float(x) for x in elements[:10]]
    explicits[4] = explicits[4] * explicits[4]
    explicits[5] = explicits[5] * explicits[5]
    explicits[6] = explicits[6] * explicits[6]
    # log.info(elements[10])
    if verbose:
      symmetry = bool(int(elements[10]))
      log.info(f"Row {idx} {'is' if symmetry else 'is not'} symmetric.")
    implicits = [float(x) for x in elements[11:]]
    if verbose:
      has_implicits = bool(implicits)
      log.info(
          f"Row {idx} {'has' if has_implicits else 'does not have'} implicits.")
    # TODO(kgenova) Validate the SIF embedding matches the expected symmetry.
    rep.append(explicits + implicits)
  rep = np.array(rep, dtype=np.float32)
  if verbose:
    log.info(f'Representation shape is {rep.shape}')
  return rep


def read_lines(p):
  with base_util.FS.open(p, 'rt') as f:
    contents = f.read()
    split_s = '\n'
    ls = [x for x in contents.split(split_s) if x]
  return ls


def read_image(p):
  """Reads in an image file that PIL.Image supports and converts to an array."""
  with base_util.FS.open(p, 'rb') as f:
    arr = np.array(Image.open(f), dtype=np.float32)
  return arr


def read_npz(p):
  if p[-4:] != '.npz':
    raise ValueError(f'Expected .npz ending for file {p}.')
  with base_util.FS.open(p, 'rb') as f:
    arr = dict(np.load(f, allow_pickle=True))
  return arr


def read_np(p):
  if p[-4:] != '.npy':
    raise ValueError(f'Expected .npy ending for file {p}.')
  with base_util.FS.open(p, 'rb') as f:
    arr = np.load(f)
  return arr


def read_txt_to_np(p):
  with base_util.FS.open(p, 'rt') as f:
    return np.loadtxt(f)


def read_py2_pkl(p):
  if p[-4:] != '.pkl':
    raise ValueError(f'Expected .pkl ending for file {p}.')
  with base_util.FS.open(p, 'rb') as f:
    # pkl = dict(np.load(f, allow_pickle=True))
    pkl = pickle.load(f, encoding='latin1')
  return pkl


def write_mesh(path, mesh):
  mesh_str = mesh_util.serialize(mesh)
  with base_util.FS.open(path, 'wb') as f:
    f.write(mesh_str)


def read_mesh(path):
  with base_util.FS.open(path, 'rb') as f:
    mesh_str = f.read()
  return mesh_util.deserialize(mesh_str)


def read_csv(path):
  with base_util.FS.open(path, 'rt') as f:
    df = pd.read_csv(f)
  return df


def read_normals(path_to_dir, im_count=20, leading='_depth'):
  """Loads gaps normals files from a conf2img output directory."""
  # Now load the files:
  out = []
  for i in range(im_count):
    base = f'{path_to_dir}/{str(i).zfill(6)}{leading}'
    paths = [base + ext for ext in ['_nx.png', '_ny.png', '_nz.png']]
    ns = [read_image(path) / 32768.0 - 1.0 for path in paths]
    normals = np.stack(ns, axis=-1)
    # We need to renormalize but the background is zero, can't divide by that:
    nrm = np.linalg.norm(normals, axis=-1, keepdims=True)
    is_background = np.squeeze(nrm > 1.1)
    normals /= np.maximum(nrm, 1e-10)
    normals[is_background, :] = 0.0
    out.append(normals)
  return np.stack(out).astype(np.float32)


def write_points(path_ext_optional, points):
  """Writes a pointcloud in the most appropriate GAPS format.

  Args:
    path_ext_optional: String. The path for the file to write. A file extension
      is optional. If there is a file extension, it must match the data
      dimensionality.
    points: Numpy array with shape [point_count, 3/4/6]. A set of XYZ points,
      optionally with a weight/value or normals (both simultaneously is not
      supported).

  Returns:
    The path to the written file, with an extension.
  """
  has_df = points.shape[-1] == 4
  has_ext = path_ext_optional[-4] == '.'
  path_no_ext, ext = os.path.splitext(path_ext_optional)
  has_ext = bool(ext)
  if has_df:
    ptspath = path_no_ext + '.sdf'
    if has_ext:
      assert ext == '.sdf'
    points = points.reshape([-1, 4])
    with base_util.FS.open(ptspath, 'wb') as f:
      # GAPS expects the floats serialized to disk to be 32-bit:
      points = points.astype(np.float32)
      f.write(points.tobytes())
  else:
    ptspath = path_no_ext + '.pts'
    if has_ext:
      assert ext == '.pts'
    has_normals = points.shape[-1] == 6
    if not has_normals:
      points = points[..., :3]
      normals = np.zeros_like(points)
      points = np.concatenate([points, normals], axis=-1)
    points = np.reshape(points, [-1, 6])
    with base_util.FS.open(ptspath, 'wb') as f:
      f.write(points.astype(np.float32).tobytes())
  return ptspath


def write_grd(path, volume, world2grid=None):
  """Writes a GAPS .grd file containing a voxel grid and world2grid matrix."""
  volume = np.squeeze(volume)
  assert len(volume.shape) == 3
  header = [int(s) for s in volume.shape]
  if world2grid is not None:
    header += [x.astype(np.float32) for x in np.reshape(world2grid, [16])]
    log.info('header: ', repr(header))
  else:
    header += [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
  header = struct.pack(3*'i' + 16*'f', *header)
  content = volume.astype('f').tostring()
  with base_util.FS.open(path, 'wb') as f:
    f.write(header)
    f.write(content)


def write_depth_image(path, depth_image):
  depth_image = (depth_image * 1000).astype(np.uint16)
  array_buffer = depth_image.tobytes()
  img = Image.new('I', depth_image.T.shape)
  img.frombytes(array_buffer, 'raw', 'I;16')
  with base_util.FS.open(path, 'wb') as f:
    img.save(f, format='PNG')

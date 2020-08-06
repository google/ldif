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
"""Inference utility functions."""

import os

import time

import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import file_util
from ldif.util import path_util
# pylint: enable=g-bad-import-order

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
}
cat_to_synset = {v:k for k,v in synset_to_cat.items()}


def parse_xid_str(xidstr):
  xids = []
  tokens = xidstr.split(',')
  for token in tokens:
    try:
      xids.append(int(token))
    except:
      raise ValueError('Could not parse %s as an xid from string %s.' %
                       (token, xidstr))
  return xids


def ensure_split_valid(split):
  valid_splits = ['train', 'test', 'val']
  if split not in valid_splits:
    raise ValueError('Unrecognized split: %s not in %s.' %
                     (split, repr(valid_splits)))


def ensure_category_valid(category):
  valid_categories = ['bodyshapes', 'all', 'airplane', 'bench', 'cabinet',
                      'car', 'chair', 'display', 'lamp', 'speaker', 'rifle',
                      'sofa', 'table', 'telephone', 'watercraft']
  if category not in valid_categories:
    raise ValueError('Unrecognized category: %s not in %s.' %
                     (category, repr(valid_categories)))


def ensure_synset_valid(synset):
  if synset not in synset_to_cat:
    raise ValueError('Unrecognized synset: %s' % repr(synset))


def ensure_hash_valid(h):
  """This does not guarantee only hashes get through, it's best-effort."""
  passes = True
  if not isinstance(h, str):
    passes = False
  elif [x for x in h if x not in '0123456789abcdef-u']:
    passes = False
  if not passes:
    raise ValueError('Invalid hash: %s' % repr(h))


def parse_xid_to_ckpt(xid_to_ckpt):
  tokens = [int(x) for x in xid_to_ckpt.split(',')]
  xids = tokens[::2]
  ckpts = tokens[1::2]
  return {k: v for (k, v) in zip(xids, ckpts)}


def get_npz_paths(split, category, modifier=''):
  """Returns the list of npz paths for the split's ground truth."""
  t = time.time()
  ensure_split_valid(split)
  filelist = os.path.join(
      path_util.get_path_to_ldif_root(),
      'data/basedirs/%s-%s%s.txt' % (split, category, modifier))
  try:
    filenames = file_util.readlines(filelist)
  except:
    raise ValueError('Unable to read filelist %s.' % filelist)
  tf.logging.info('Loaded filelist in %0.2fms.', (time.time() - t))
  return filenames


def get_mesh_identifiers(split, category):
  if category == 'unseen':
    raise NotImplementedError('No longer supported.')
  npz_paths = get_npz_paths(split, category)
  out = []
  for p in npz_paths:
    sp, sy, mh = p.split('/')[-3:]
    mh = mh.replace('.npz', '')
    out.append('/'.join([sp, sy, mh]))
  return out


def get_rgb_paths(split, category, modifier=''):
  """Reads a file containing a list of .npz files on which to run inference."""
  t = time.time()
  ensure_split_valid(split)
  ensure_category_valid(category)
  path = ('/ROOT_DIR/%s-%s%s.txt') % (category, split, modifier)
  try:
    files = file_util.readlines(path)
  except:
    raise ValueError('Failed to read filelist %s' % path)
  tf.logging.info('Time loading filelist %0.2f', (time.time() - t))
  return files


def rgb_path_to_synset_and_hash(rgb_path):
  synset, mesh_hash = rgb_path.split('/')[-4:-2]
  ensure_synset_valid(synset)
  ensure_hash_valid(mesh_hash)
  return synset, mesh_hash


def rgb_path_to_npz_path(rgb_path, split, dataset='shapenet-occnet'):
  synset, mesh_hash = rgb_path_to_synset_and_hash(rgb_path)
  npz_path = 'ROOT_DIR/%s/%s/%s/%s.npz' % (
      dataset, split, synset, mesh_hash)
  return npz_path


def parse_npz_path(path):
  """Extracts the split, synset, and mesh hash from an npz path."""
  split, synset, npz = path.split('/')[-3:]
  if '.npz' not in npz:
    raise ValueError('Error parsing hash from %s' % path)
  if synset not in synset_to_cat:
    raise ValueError('Error parsing synset from %s' % path)
  try:
    ensure_split_valid(split)
  except ValueError:
    raise ValueError('Error parsing split from %s' % path)
  mesh_hash = npz.replace('.npz', '')
  return split, synset, mesh_hash


def read_png_to_float_npy_with_reraising(path):
  try:
    im = file_util.read_image(path) / 255.0
  except ValueError:
    tf.logging.info('Failed to load file %s', path)
    raise ValueError('Failed to load file %s' % path)
  return im

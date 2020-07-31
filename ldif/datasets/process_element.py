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
"""A local tf.Dataset wrapper for LDIF."""

import os
import sys
import time

import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.inference import example
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order


def load_example_dict(example_directory, log_level=None):
  """Loads an example from disk and makes a str:numpy dictionary out of it."""
  if log_level:
    log.set_level(log_level)
  entry_t = time.time()
  start_t = entry_t  # Keep the function entry time around for a cumulative print.
  e = example.InferenceExample.from_directory(example_directory, verbose=False)
  end_t = time.time()
  log.verbose(f'Make example: {end_t - start_t}')
  start_t = end_t

  # The from_directory method should probably optionally take in a synset.
  bounding_box_samples = e.uniform_samples
  end_t = time.time()
  log.verbose(f'Bounding box: {end_t - start_t}')
  start_t = end_t
  # TODO(kgenova) There is a pitfall here where the depth is divided by 1000,
  # after this. So if some other depth images are provided, they would either
  # need to also be stored in the GAPS format or be artificially multiplied
  # by 1000.
  depth_renders = e.depth_images  # [20, 224, 224, 1]. 1 or 1000? trailing 1?
  assert depth_renders.shape[0] == 1
  depth_renders = depth_renders[0, ...]
  end_t = time.time()
  log.verbose(f'Depth renders: {end_t - start_t}')
  start_t = end_t

  mesh_name = e.mesh_name
  end_t = time.time()
  log.verbose(f'Mesh name: {end_t - start_t}')
  start_t = end_t

  log.verbose(f'Loading {mesh_name} from split {e.split}')
  near_surface_samples = e.near_surface_samples
  end_t = time.time()
  log.verbose(f'NSS: {end_t - start_t}')

  start_t = end_t
  grid = e.grid
  end_t = time.time()
  log.verbose(f'Grid: {end_t - start_t}')
  start_t = end_t

  world2grid = e.world2grid
  end_t = time.time()
  log.verbose(f'world2grid: {end_t - start_t}')
  start_t = end_t

  surface_point_samples = e.precomputed_surface_samples_from_dodeca
  end_t = time.time()
  log.verbose(f'surface points: {end_t - start_t}')
  log.verbose(f'load_example_dict total time: {end_t - entry_t}')
  return {
      'bounding_box_samples': bounding_box_samples,
      'depth_renders': depth_renders,
      'mesh_name': mesh_name,
      'near_surface_samples': near_surface_samples,
      'grid': grid,
      'world2grid': world2grid,
      'surface_point_samples': surface_point_samples,
  }


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))


def _bytes_feature(value):
  if isinstance(value, str):
    value = value.encode('utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_tf_example(d):
  feature = {
      'bounding_box_samples': _float_feature(d['bounding_box_samples']),
      'depth_renders': _float_feature(d['depth_renders']),
      'mesh_name': _bytes_feature(d['mesh_name']),
      'near_surface_samples': _float_feature(d['near_surface_samples']),
      'grid': _float_feature(d['grid']),
      'world2grid': _float_feature(d['world2grid']),
      'surface_point_samples': _float_feature(d['surface_point_samples'])
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def full_featurespec():
  return {
      'bounding_box_samples': tf.io.FixedLenFeature([100000, 4], tf.float32),
      'depth_renders': tf.io.FixedLenFeature([20, 224, 224, 1], tf.float32),
      'mesh_name': tf.io.FixedLenFeature([], tf.string),
      'near_surface_samples': tf.io.FixedLenFeature([100000, 4], tf.float32),
      'grid': tf.io.FixedLenFeature([32, 32, 32], tf.float32),
      'world2grid': tf.io.FixedLenFeature([4, 4], tf.float32),
      'surface_point_samples': tf.io.FixedLenFeature([10000, 6], tf.float32)
  }


def parse_tf_example(example_proto):
  d = tf.io.parse_single_example(example_proto, full_featurespec())
  return (d['bounding_box_samples'], d['depth_renders'], d['mesh_name'],
          d['near_surface_samples'], d['grid'], d['world2grid'],
          d['surface_point_samples'])


def _example_dict_tf_func_wrapper(mesh_orig_path):
  mesh_orig_path = mesh_orig_path.decode(sys.getdefaultencoding())
  assert '/mesh_orig.ply' in mesh_orig_path
  example_directory = mesh_orig_path.replace('/mesh_orig.ply', '')
  d = load_example_dict(example_directory)
  return (d['bounding_box_samples'], d['depth_renders'], d['mesh_name'],
          d['near_surface_samples'], d['grid'], d['world2grid'],
          d['surface_point_samples'])


def parse_example(filename):
  """A tensorflow function to return a dataset element when mapped"""
  return tf.py_func(_example_dict_tf_func_wrapper, [filename], [
          tf.float32, tf.float32, tf.string, tf.float32, tf.float32, tf.float32,
          tf.float32])


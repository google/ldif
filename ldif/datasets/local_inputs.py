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


def load_example_dict(example_directory):
  """Loads an example from disk and makes a str:numpy dictionary out of it."""
  entry_t = time.time()
  start_t = entry_t  # Keep around the function entry time for a cumulative print.
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


def _example_dict_tf_func_wrapper(mesh_orig_path):
  # log.info(f'The input path is {mesh_orig_path}')
  mesh_orig_path = mesh_orig_path.decode(sys.getdefaultencoding())
  assert '/mesh_orig.ply' in mesh_orig_path
  example_directory = mesh_orig_path.replace('/mesh_orig.ply', '')
  d = load_example_dict(example_directory)
  return (d['bounding_box_samples'], d['depth_renders'], d['mesh_name'],
          d['near_surface_samples'], d['grid'], d['world2grid'],
          d['surface_point_samples'])


def parse_example(ex):
  log.info(f'The input example is: {ex}.')
  # TODO(kgenova) We can call a py_func here that will actually build the object
  # from the filename. But that will be subject to the GIL, so maybe it's better
  # to just make tfrecords ahead of time? Though tfrecords will have a space
  # limitation, at least for single-view depth. So best to write the function
  # and see whether there's an IO
  # bottleneck, and deal with it if there is. There are also in-between
  # solutions that might be better, like preprocessing the data in such a way
  # that observation + 3D exist and can be dynamically combined only in terms
  # of pure python.
  # Either way, the place to start is a function that goes from a filename to
  # a dictionary (?) with the necessary inputs.


def make_dataset(directory, batch_size, mode, split):
  """Generates a one-shot style tf.Dataset."""
  assert split in ['train', 'val', 'test']
  dataset = tf.data.Dataset.list_files(f'{directory}/{split}/*/*/mesh_orig.ply')
  log.info('Mapping...')
  if mode == 'train':
    dataset = dataset.shuffle(buffer_size=2 * batch_size)
    dataset = dataset.repeat()
  # pylint: disable=g-long-lambda
  dataset = dataset.map(
      lambda filename: tf.py_func(_example_dict_tf_func_wrapper, [filename], [
          tf.float32, tf.float32, tf.string, tf.float32, tf.float32, tf.float32,
          tf.float32
      ]),
      num_parallel_calls=os.cpu_count())
  # pylint: enable=g-long-lambda

  bs = batch_size
  dataset = dataset.batch(bs, drop_remainder=True).prefetch(1)

  dataset_items = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
  dataset_obj = lambda: 0
  dataset_obj.bounding_box_samples = tf.ensure_shape(dataset_items[0],
                                                     [bs, 100000, 4])
  dataset_obj.depth_renders = tf.ensure_shape(dataset_items[1],
                                              [bs, 20, 224, 224, 1])
  dataset_obj.mesh_name = dataset_items[2]
  dataset_obj.near_surface_samples = tf.ensure_shape(dataset_items[3],
                                                     [bs, 100000, 4])
  dataset_obj.grid = tf.ensure_shape(dataset_items[4], [bs, 32, 32, 32])
  dataset_obj.world2grid = tf.ensure_shape(dataset_items[5], [bs, 4, 4])
  dataset_obj.surface_point_samples = tf.ensure_shape(dataset_items[6],
                                                      [bs, 10000, 6])

  return dataset_obj

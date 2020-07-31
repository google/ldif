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
import glob
import sys
import time

import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.datasets import process_element
from ldif.inference import example
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order

def _make_optimized_dataset(directory, batch_size, mode, split):
  filenames = glob.glob(f'{directory}/optimized/{split}/*.tfrecords')
  log.verbose(f'Making dataset from the following files: {filenames}')
  dataset = tf.data.TFRecordDataset(filenames=filenames, compression_type='GZIP',
        buffer_size=None, num_parallel_reads=8)
  log.info('Mapping...')
  if mode == 'train':
    dataset = dataset.shuffle(buffer_size=2 * batch_size)
    dataset = dataset.repeat()

  dataset = dataset.map(process_element.parse_tf_example,
      num_parallel_calls=os.cpu_count())

  dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

  dataset_items = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
  return build_dataset_obj(dataset_items, batch_size)



def build_dataset_obj(dataset_items, bs):
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




def make_dataset(directory, batch_size, mode, split):
  """Generates a one-shot style tf.Dataset."""
  assert split in ['train', 'val', 'test']
  # Detect if an optimized dataset exists:
  if os.path.isdir(f'{directory}/optimized'):
    log.info(f'Optimized dataset detected at {directory}/optimized')
    return _make_optimized_dataset(directory, batch_size, mode, split)
  log.info(f'No optimized preprocessed dataset found at {directory}/optimized. '
          'Processing dataset elements on the fly. If an IO bottleneck is '
          'present, please rerun meshes2dataset with --optimize.')

  dataset = tf.data.Dataset.list_files(f'{directory}/{split}/*/*/mesh_orig.ply')
  log.info('Mapping...')
  if mode == 'train':
    dataset = dataset.shuffle(buffer_size=2 * batch_size)
    dataset = dataset.repeat()

  dataset = dataset.map(process_element.parse_example,
      num_parallel_calls=os.cpu_count())

  bs = batch_size
  dataset = dataset.batch(bs, drop_remainder=True).prefetch(1)

  dataset_items = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
  return build_dataset_obj(dataset_items, bs)

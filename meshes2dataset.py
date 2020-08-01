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
"""This script takes a directory of meshes and generates a (D)SIF dataset.

The dataset can be used for training, evaluation, and inference on ldif models.
"""

import glob
import random
import os

from absl import app
from absl import flags

# pylint: disable=g-multiple-import
from joblib import Parallel, delayed
# pylint: enable=g-multiple-import

import tqdm
import tensorflow as tf

# LDIF is local code, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.datasets import process_element
from ldif.scripts import make_example
from ldif.util import file_util
from ldif.util import path_util
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order

FLAGS = flags.FLAGS

flags.DEFINE_string('mesh_directory', '', 'Path to meshes. This folder should'
                    ' have the structure <root>/{train,test,val}/<class>/*.ply')

flags.DEFINE_string('dataset_directory', '', 'Path to output dataset.')

flags.DEFINE_boolean(
    'skip_existing', True, 'Whether to skip process examples'
    ' that are already written into the output dataset. True'
    ' enables completing a processing run that crashed, or '
    ' adding examples to a dataset that already exists with'
    ' partial overlap. False enables updating a dataset'
    ' in-place.')

flags.DEFINE_integer(
    'max_threads', -1, 'The maximum number of threads to use.'
    ' If -1, will allocate all available threads on CPU.')

flags.DEFINE_string('log_level', 'INFO',
    'One of VERBOSE, INFO, WARNING, ERROR. Sets logs to print '
    'only at or above the specified level.')

flags.DEFINE_boolean(
    'optimize', True, 'Whether to create an optimized tfrecords '
    'dataset. This will substantially improve IO throughput, at '
    'the expense of approximately doubling disk usage and adding '
    'a moderate amount of additional dataset creation time. '
    'Recommended unless disk space is very tight or data is stored '
    'on a local NVMe drive or similar.')

flags.DEFINE_boolean(
    'trample_optimized', True, 'Whether to erase and re-create '
    'optimized files. Set True if changes have been made to the '
    'dataset since the last time meshes2dataset was run; set '
    'False to complete optimization if it was halted midway.')

flags.DEFINE_boolean(
    'optimize_only', False, 'Whether to skip dataset creation '
    'and only write tfrecords files.')


def process_one(f, mesh_directory, dataset_directory, skip_existing, log_level):
  """Processes a single mesh, adding it to the dataset."""
  relpath = f.replace(mesh_directory, '')
  assert relpath[0] == '/'
  relpath = relpath[1:]
  split, synset = relpath.split('/')[:2]
  log.verbose(f'The split is {split} and the synset is {synset}')
  name = os.path.basename(f)
  name, extension = os.path.splitext(name)
  valid_extensions = ['.ply']
  if extension not in valid_extensions:
    raise ValueError(f'File with unsupported extension {extension} found: {f}.'
                     f' Only {valid_extensions} are supported.')
  output_dir = f'{dataset_directory}/{split}/{synset}/{name}/'
  # This is the last file the processing writes, if it already exists the
  # example has already been processed.
  final_file_written = f'{output_dir}/depth_and_normals.npz'
  make_example.mesh_to_example(
      os.path.join(path_util.get_path_to_ldif_parent(), 'ldif'), f,
      f'{dataset_directory}/{split}/{synset}/{name}/', skip_existing, log_level)
  return output_dir

def serialize(example_dir, log_level):
  d = process_element.load_example_dict(example_dir, log_level)
  s = process_element.make_tf_example(d)
  return s

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  random.seed(2077)
  log.set_level(FLAGS.log_level)

  n_jobs = os.cpu_count()
  assert FLAGS.max_threads != 0
  if FLAGS.max_threads > 0:
    n_jobs = FLAGS.max_threads

  mesh_directory = FLAGS.mesh_directory
  if mesh_directory[-1] == '/':
    mesh_directory = mesh_directory[:-1]

  files = glob.glob(f'{mesh_directory}/*/*/*.ply')

  if not files and not FLAGS.optimize_only:
    raise ValueError(f"Didn't find any ply files in {mesh_directory}. "
                     "Please make sure the directory structure is "
                     "[mesh_directory]/[splits]/[class names]/[ply files]")

  # Make the directories first because it's not threadsafe and also might fail.
  if files and not FLAGS.optimize_only:
    log.info('Creating directories...')
    for i, f in tqdm.tqdm(enumerate(files)):
      relpath = f.replace(mesh_directory, '')
      # log.info(f'Relpath: {relpath}')
      assert relpath[0] == '/'
      relpath = relpath[1:]
      split, synset = relpath.split('/')[:2]
      if not os.path.isdir(f'{FLAGS.dataset_directory}/{split}'):
        os.makedirs(f'{FLAGS.dataset_directory}/{split}')
      if not os.path.isdir(f'{FLAGS.dataset_directory}/{split}/{synset}'):
        os.mkdir(f'{FLAGS.dataset_directory}/{split}/{synset}')
    log.info('Making dataset...')
    # Flags can't be pickled:
    output_dirs = Parallel(n_jobs=n_jobs)(
        delayed(process_one)(f, mesh_directory, FLAGS.dataset_directory,
                             FLAGS.skip_existing, FLAGS.log_level) for f in tqdm.tqdm(files))
    log.info('Making dataset registry...')
  else:
    output_dirs = glob.glob(f'{FLAGS.dataset_directory}/*/*/*/surface_samples_from_dodeca.pts')
    output_dirs = [os.path.dirname(f) + '/' for f in output_dirs]
  output_dirs.sort()  # So randomize with a fixed seed always results in the same order
  splits = {x.split('/')[-4] for x in output_dirs}
  if 'optimized' in splits:
    raise ValueError(f'The keyword "optimized" cannot be used for a split name, it is reserved.')
  for split in splits:
    elements_of_split = [x for x in output_dirs if x.split('/')[-4] == split]
    with open(f'{FLAGS.dataset_directory}/{split}.txt', 'wt') as f:
      f.write('\n'.join(elements_of_split) + '\n')
  log.info('Done!')

  if FLAGS.optimize:
    log.info('Precomputing optimized tfrecord files...')
    opt_dir = f'{FLAGS.dataset_directory}/optimized'
    if FLAGS.trample_optimized and os.path.isdir(opt_dir):
      for f in os.listdir(opt_dir):
        if f.endswith('.tfrecords'):
          os.remove(os.path.join(opt_dir, f))
    if not os.path.isdir(opt_dir):
      os.mkdir(opt_dir)
    for split in splits:
      log.info(f'Optimizing split {split}...')
      elements_of_split = [x for x in output_dirs if x.split('/')[-4] == split]
      examples_per_shard=64
      # Make sure shards are totally random:
      random.shuffle(elements_of_split)
      n_shards = int(len(elements_of_split) / examples_per_shard)
      if len(elements_of_split) % examples_per_shard:
        n_shards += 1
      shard_dir = f'{FLAGS.dataset_directory}/optimized/{split}'
      if not os.path.isdir(shard_dir):
        os.mkdir(shard_dir)
      for shard_idx in tqdm.tqdm(range(n_shards)):
        shard_name = f'{shard_dir}/{split}-%.5d-of-%.5d.tfrecords' % (shard_idx, n_shards)
        if not FLAGS.trample_optimized and os.path.isfile(shard_name):
          continue
        start_idx = shard_idx * examples_per_shard
        end_idx = (shard_idx + 1) * examples_per_shard
        options = tf.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.GZIP)
        with tf.io.TFRecordWriter(shard_name, options=options) as writer:
          to_process = elements_of_split[start_idx:end_idx]
          serialized = Parallel(n_jobs=n_jobs)(delayed(serialize)(d, FLAGS.log_level)
                for d in to_process)
          for s in serialized:
            writer.write(s)


if __name__ == '__main__':
  app.run(main)

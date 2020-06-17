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
import os

from absl import app
from absl import flags

# pylint: disable=g-multiple-import
from joblib import Parallel, delayed
# pylint: enable=g-multiple-import

import tqdm

# LDIF is local code, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.scripts import make_example
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


def process_one(f, mesh_directory, dataset_directory, skip_existing):
  """Processes a single mesh, adding it to the dataset."""
  relpath = f.replace(mesh_directory, '')
  assert relpath[0] == '/'
  relpath = relpath[1:]
  split, synset = relpath.split('/')[:2]
  log.info(f'The split is {split} and the synset is {synset}')
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
      f'{dataset_directory}/{split}/{synset}/{name}/', skip_existing)
  return output_dir


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  mesh_directory = FLAGS.mesh_directory
  if mesh_directory[-1] == '/':
    mesh_directory = mesh_directory[:-1]

  files = glob.glob(f'{mesh_directory}/*/*/*.ply')

  if not files:
    raise ValueError(f"Didn't find any ply files in {mesh_directory}")

  # Make the directories first because it's not threadsafe and also might fail.
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
  n_jobs = os.cpu_count()
  assert FLAGS.max_threads != 0
  if FLAGS.max_threads > 0:
    n_jobs = FLAGS.max_threads
  output_dirs = Parallel(n_jobs=n_jobs)(
      delayed(process_one)(f, mesh_directory, FLAGS.dataset_directory,
                           FLAGS.skip_existing) for f in tqdm.tqdm(files))
  log.info('Making dataset registry...')
  splits = {x.split('/')[-4] for x in output_dirs}
  for split in splits:
    elements_of_split = [x for x in output_dirs if x.split('/')[-4] == split]
    with open(f'{FLAGS.dataset_directory}/{split}.txt', 'wt') as f:
      f.write('\n'.join(elements_of_split) + '\n')
  log.info('Done!')


if __name__ == '__main__':
  app.run(main)

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
"""Evaluates a trained LDIF/SIF model."""

import os
import random
import time

from absl import app
from absl import flags

import pandas as pd
# Imports have to be in this order to silence tensorflow:
# pylint: disable=g-import-not-at-top
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tqdm

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.inference import predict
from ldif.util import file_util
from ldif.util import gaps_util
from ldif.util import gpu_util
from ldif.util import path_util
from ldif.inference import example as examples
from ldif.inference import metrics
from ldif.inference import util
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order
# pylint: enable=g-import-not-at-top

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'dataset_directory', '', 'The path to the dataset to'
    'evaluate (should be the same as the dataset_directory flag'
    ' passed to meshes2dataset.py to create the dataset.')

flags.DEFINE_string(
    'result_directory', '', 'The directory where result files'
    ' should be written. Only necessary if one of the save_'
    ' flags is set to true (all are false by default).')

flags.DEFINE_boolean(
    'save_meshes', False, 'If true, the output meshes will'
    ' be written to the result directory with a structure mirroring'
    ' the dataset directory.')

flags.DEFINE_boolean(
    'save_ldifs', False, 'If provided, the output ldif.txt files'
    ' will be written to the result directory with a structure'
    ' mirroring the dataset directory. Should not be the same'
    ' directory as save_meshes_to if provided.')

flags.DEFINE_boolean(
    'save_results', False, 'If provided, two CSV files will be written to the'
    ' result directory. The first will contain the mean results over each class'
    ', while the second will contain the results for every mesh in the'
    ' dataset split.')

flags.DEFINE_boolean(
    'use_gpu_for_tensorflow', True, 'Whether to enable use of'
    ' the GPU by tensorflow. Set to false by default because'
    ' running eval while training is common, but not possible'
    ' if only a single GPU is available and the eval job'
    ' needs it, because tensorflow allocates most GPU memory.'
    ' Note that regardless of this setting, if the inference'
    ' mode is set to use the custom CUDA kernel, that will'
    ' use a gpu anyway (it can be separately disabled).'
    ' However, disabling that is typically not necessary'
    ' because it uses only a very a small amount of VRAM.')

flags.DEFINE_boolean(
    'use_inference_kernel', True, 'Whether to enable use'
    ' of the custom CUDA kernel for LDIF inference. Note that'
    ' to be used, it must first be compiled (this step should'
    ' hopefully be easy, see the README for more details).'
    ' The speed increase should be several orders of magnitude'
    ', so it is highly recommended.')

flags.DEFINE_string('experiment_name', 'reproduce-ldif',
                    'The name of the experiment to'
                    ' evaluate')

flags.DEFINE_integer(
    'ckpt', -1, 'The index of the checkpoint to evaluate. If'
    ' -1, then evaluates the most recent checkpoint.')

flags.DEFINE_string('split', 'test', 'The split(s) to evaluate, comma separated.')

flags.DEFINE_float(
    'eval_frac', 1.0, 'The fraction of the dataset to evaluate.'
    ' If the fraction is less than 1, then the subset will be'
    ' chosen randomly.')

flags.DEFINE_boolean(
    'compute_metrics', True, 'If false, the model will be'
    ' run, but metrics will not be computed.')

flags.DEFINE_string(
    'model_directory', 'trained_models/', 'The path to the trained model root'
    ' directory. Can be absolute or relative to the LDIF repository root.')

flags.DEFINE_boolean('visualize', False,
                     'If true, interactively visualizes each reconstruction.')

flags.DEFINE_string(
    'log_level', 'INFO',
    'One of VERBOSE, INFO, WARNING, ERROR. Sets logs to print '
    'only at or above the specified level.')

flags.DEFINE_integer('resolution', 256,
                     'The resolution at which to do marching cubes.')

flags.DEFINE_string(
    'only_class', '', 'Only evaluate on this class, if provided.')


def get_model_root():
  """Finds the path to the trained model's root directory based on flags."""
  ldif_abspath = path_util.get_path_to_ldif_root()
  model_dir_is_relative = FLAGS.model_directory[0] != '/'
  if model_dir_is_relative:
    model_dir_path = os.path.join(ldif_abspath, FLAGS.model_directory)
  else:
    model_dir_path = FLAGS.model_directory
  if not os.path.isdir(model_dir_path):
    raise ValueError(f'Could not find model directory {model_dir_path}')
  return model_dir_path


def load_newest_model():
  """Loads the newest checkpoint of the specified model."""
  model_root = get_model_root()
  model_name = 'sif-transcoder'
  experiment_name = FLAGS.experiment_name
  encoder = predict.DepthEncoder.from_modeldir(
      model_root, model_name, experiment_name, xid=1, ckpt_idx=-1)
  decoder = predict.Decoder.from_modeldir(
      model_root, model_name, experiment_name, xid=1, ckpt_idx=-1)
  decoder.use_inference_kernel = FLAGS.use_inference_kernel
  return encoder, decoder


def get_evaluation_directories(split):
  registry_path = f'{FLAGS.dataset_directory}/{split}.txt'
  items_to_eval = file_util.readlines(registry_path)
  return items_to_eval


def filter_by_eval_frac(items):
  tmp = [x for x in items]
  random.shuffle(tmp)
  to_keep = int(len(tmp) * FLAGS.eval_frac)
  to_keep = max(1, to_keep)
  return tmp[:to_keep]


def filter_by_class(items):
  if not FLAGS.only_class:
    return items
  class_or_synset = FLAGS.only_class
  if class_or_synset in util.cat_to_synset:
    key = util.cat_to_synset[class_or_synset]
  else:
    key = class_or_synset
  
  class_items = [x for x in items if f'/{key}/' in x]
  if not class_items:
    raise ValueError(f'Filtering by class {key} results in no elements.')
  return class_items


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  log.set_level(FLAGS.log_level)
  tf.disable_v2_behavior()

  gpu_util.get_free_gpu_memory(0)
  if FLAGS.use_gpu_for_tensorflow and FLAGS.use_inference_kernel:
    log.info('Limiting TensorFlow memory by 1GB so the inference kernel'
             ' has enough left over to run.')

  if not FLAGS.dataset_directory:
    raise ValueError('A dataset directory must be provided.')
  if not FLAGS.result_directory:
    if FLAGS.save_results or FLAGS.save_meshes or FLAGS.save_ldifs:
      raise ValueError('A result directory must be provided to save results.')
  else:
    if not os.path.isdir(FLAGS.result_directory):
      os.makedirs(FLAGS.result_directory)
  if not FLAGS.use_gpu_for_tensorflow:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

  log.info('Loading model...')
  # Try to detect the most common error early for a good warning message:
  if not os.path.isdir(get_model_root()):
    raise ValueError(f"Couldn't find a trained model at {get_model_root()}")
  encoder, decoder = load_newest_model()

  log.info('Evaluating metrics...')
  splits = [x for x in FLAGS.split.split(',') if x]
  log.info(f'Will evaluate on splits: {splits}')
  for split in splits:
    log.info(f'Starting evaluation for split {split}.')
    dataset_items = get_evaluation_directories(split)
    log.info(f'The split has {len(dataset_items)} elements.')
    results = []
    to_eval = filter_by_class(dataset_items)
    to_eval = filter_by_eval_frac(to_eval)
    for path in tqdm.tqdm(to_eval):
      e = examples.InferenceExample.from_directory(path)
      embedding = encoder.run_example(e)
      iou = decoder.iou(embedding, e)
      gt_mesh = e.gt_mesh
      mesh = decoder.extract_mesh(embedding, resolution=FLAGS.resolution)
      if FLAGS.visualize:
        # Visualize in the normalized_coordinate frame, so the camera is
        # always reasonable. Metrics are computed in the original frame.
        gaps_util.mshview([e.normalized_gt_mesh, mesh])
  
      # TODO(kgenova) gaps2occnet is poorly named, it is really normalized ->
      # unnormalized (where 'gaps' is the normalized training frame and 'occnet'
      # is whatever the original frame of the input mesh was)
      post_extract_start = time.time()
      mesh.apply_transform(e.gaps2occnet)
  
      if FLAGS.save_meshes:
        path = (f'{FLAGS.result_directory}/meshes/{split}/{e.cat}/'
                f'{e.mesh_hash}.ply')
        if not os.path.isdir(os.path.dirname(path)):
          os.makedirs(os.path.dirname(path))
        mesh.export(path)
      if FLAGS.save_ldifs:
        path = (f'{FLAGS.result_directory}/ldifs/{split}/{e.cat}/'
                f'{e.mesh_hash}.txt')
        if not os.path.isdir(os.path.dirname(path)):
          os.makedirs(os.path.dirname(path))
        decoder.savetxt(embedding, path)
  
      nc, fst, fs2t, chamfer = metrics.all_mesh_metrics(mesh, gt_mesh)
      log.verbose(f'Mesh: {e.mesh_name}')
      log.verbose(f'IoU: {iou}.')
      log.verbose(f'F-Score (tau): {fst}')
      log.verbose(f'Chamfer: {chamfer}')
      log.verbose(f'F-Score (2*tau): {fs2t}')
      log.verbose(f'Normal Consistency: {nc}')
      results.append({
          'key': e.mesh_name,
          'Normal Consistency': nc,
          'F-Score (tau)': fst,
          'F-Score (2*tau)': fs2t,
          'Chamfer': chamfer,
          'IoU': iou
      })
      post_extract_end = time.time()
      log.verbose(f'Time post extract: {post_extract_end - post_extract_start}')
    results = pd.DataFrame(results)
    if FLAGS.save_results:
      complete_csv = results.to_csv()
      result_path = f'{FLAGS.result_directory}/full_results_{split}.csv'
      file_util.writetxt(result_path, complete_csv)
    final_results = metrics.aggregate_extracted(results)
    if FLAGS.save_results:
      summary_out_path = f'{FLAGS.result_directory}/result_summary_{split}.csv'
      file_util.writetxt(summary_out_path, final_results.to_csv())


if __name__ == '__main__':
  app.run(main)

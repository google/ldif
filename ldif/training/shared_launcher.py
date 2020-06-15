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
"""Launcher functionality that is shared between local and remote training."""

from absl import flags

import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.training import eval_step
from ldif.training import loss
from ldif.model import model
from ldif.datasets import preprocess
from ldif.training import summarize
# pylint: enable=g-bad-import-order

from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import training as contrib_training


FLAGS = flags.FLAGS
# Assumptions the new code makes:
#  1) There are a fixed number of shape elements.
#  1) A constant, not a quadric surface.
#  3) The shape elements are predicted together, not sequentially (more
#       low-level: there is no sequential dependency of prediction K-N on
#       prediction K, no hierarchy, etc.)
#  4) We never normalize the weighted combination.
#  5) The shape elements are 3 dimensional.
#  6) The supervision is sparse (i.e. no voxel grids anywhere).
#  7) The input observation is a set of 1 or more images, each of which have
#       1 or more channels (a rank-5 tensor) I.e. no point clouds, voxels...
#  8) The network only needs to know how big each shape element is (i.e.
#     7 numbers for our current model) and how many shape elements there are.


def set_train_op(model_config):
  """Sets the train op for a single weights update."""
  with tf.name_scope('train-op-creation'):
    if model_config.hparams.opt == 'adm':
      optimizer = tf.train.AdamOptimizer(
          learning_rate=model_config.hparams.lr, beta1=0.9, beta2=0.999)
    elif model_config.hparams.opt == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=model_config.hparams.lr)
    elif model_config.hparams.opt == 'mtm':
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=model_config.hparams.lr, momentum=0.9)
    # In TPU training this wraps the optimizer in a CrossShardOptimizer for
    #   you.
    if model_config.hparams.sync == 't':
      assert model_config.hparams.gpuc > 0
      assert model_config.hparams.vbs > 0
      optimizer = tf.train.SyncReplicasOptimizer(
          optimizer,
          replicas_to_aggregate=model_config.hparams.vbs,
          total_num_replicas=model_config.hparams.gpuc)
    else:
      optimizer = model_config.wrap_optimizer(optimizer)
    variables_to_train = tf.compat.v1.trainable_variables()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if model_config.hparams.ob == 'f':
      variables_to_train = contrib_framework.filter_variables(
          variables_to_train, exclude_patterns=['explicit_embedding_cnn'])
      update_ops = contrib_framework.filter_variables(
          update_ops, exclude_patterns=['explicit_embedding_cnn'])

    model_config.train_op = contrib_training.create_train_op(
        model_config.loss,
        optimizer=optimizer,
        update_ops=update_ops,
        variables_to_train=variables_to_train,
        # transform_grads_fn=opt_util.clip_by_global_norm,
        summarize_gradients=False,
        colocate_gradients_with_ops=False)


def sif_transcoder(model_config):
  """Builds a structured implicit function transcoder.

  Args:
    model_config: A ModelConfig instance.
  """
  # Get the input data from the input_fn.
  if not model_config.train:
    model_config.hparams.bs = 1

  if model_config.hparams.rsl != 1.0:
    dataset = lambda: 0
    factor = model_config.hparams.rsl
    dataset.factor = factor
    dataset.xyz_render = model_config.inputs[
        'dataset'].xyz_render * factor
    dataset.near_surface_samples = model_config.inputs[
        'dataset'].near_surface_samples * factor
    dataset.bounding_box_samples = model_config.inputs[
        'dataset'].bounding_box_samples * factor
    xyz = model_config.inputs['dataset'].surface_point_samples[:, :, :3] * factor
    nrm = model_config.inputs['dataset'].surface_point_samples[:, :, 3:]
    dataset.surface_point_samples = tf.concat([xyz, nrm], axis=-1)
    dataset.grid = model_config.inputs['dataset'].grid * factor
    to_old_world = tf.constant(
        [[[1.0 / factor, 0.0, 0.0, 0.0], [0.0, 1.0 / factor, 0.0, 0.0],
          [0.0, 0.0, 1.0 / factor, 0.0], [0.0, 0.0, 0.0, 0.0]]],
        dtype=tf.float32)
    to_old_world = tf.tile(to_old_world, [model_config.hparams.bs, 1, 1])
    dataset.world2grid = tf.matmul(
        model_config.inputs['dataset'].world2grid, to_old_world)
    dataset.mesh_name = model_config.inputs['dataset'].mesh_name
    dataset.depth_render = model_config.inputs['dataset'].depth_render
    model_config.inputs['dataset'] = dataset

  training_example = preprocess.preprocess(model_config)

  synset = tf.strings.substr(training_example.mesh_name, len=8, pos=0)
  is_lamp = tf.math.equal(synset, '03636649')
  is_chair = tf.math.equal(synset, '03001627')
  lamp_frac = tf.reduce_mean(tf.cast(is_lamp, dtype=tf.float32))
  chair_frac = tf.reduce_mean(tf.cast(is_chair, dtype=tf.float32))
  tf.summary.scalar(f"{model_config.inputs['split']}-lamp-frac", lamp_frac)
  tf.summary.scalar(f"{model_config.inputs['split']}-chair-frac", chair_frac)

  observation = model.Observation(model_config, training_example)

  imp_net = model.StructuredImplicitModel(model_config, 'imp_net')

  prediction = imp_net.forward(observation)
  # model_config.export_signature_def_map[
  #     'autoencoder'] = prediction.export_signature_def()
  structured_implicit = prediction.structured_implicit

  if not model_config.inference:
    loss.set_loss(model_config, training_example, structured_implicit)

  if model_config.train:
    summarize.add_train_summaries(model_config, prediction)
    set_train_op(model_config)
  elif model_config.eval:
    model_config.eval_step = eval_step.make_eval_step(model_config,
                                                      training_example,
                                                      prediction)

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
"""Code for preprocessing training examples.

This code can be aware of the existence of individual datasets, but it can't be
aware of their internals.
"""

import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.datasets import shapenet
from ldif.util import random_util
# pylint: enable=g-bad-import-order


# Main entry point for preprocessing. This code uses the model config to
# generate an appropriate training example. It uses duck typing, in that
# it should dispatch generation to a dataset to handle internals, and verify
# that the training example contains the properties associated with the config.
# But it doesn't care about anything else.
def preprocess(model_config):
  """Generates a training example object from the model config."""
  # TODO(kgenova) Check if dataset is shapenet. If so, return a ShapeNet
  # training example.
  training_example = shapenet.ShapeNetExample(model_config)

  # TODO(kgenova) Look at the model config and verify that nothing is missing.
  if model_config.hparams.da == 'f':
    return training_example
  elif model_config.hparams.da == 'p':  # Pan:
    tx = random_util.random_pan_rotations(model_config.hparams.bs)
  elif model_config.hparams.da == 'r':  # SO(3):
    tx = random_util.random_rotations(model_config.hparams.bs)
  elif model_config.hparams.da == 't':
    origin = training_example.sample_sdf_near_surface(1)[0]
    origin = tf.reshape(origin, [model_config.hparams.bs, 3])
    tx = random_util.random_transformation(origin)
  elif model_config.hparams.da == 'z':
    origin = training_example.sample_sdf_near_surface(1)[0]
    origin = tf.reshape(origin, [model_config.hparams.bs, 3])
    tx = random_util.random_zoom_transformation(origin)
  training_example.apply_transformation(tx)
  if model_config.hparams.cri == 't':
    training_example.crop_input(model_config.hparams.cic)
  if model_config.hparams.clc == 't' and (model_config.train or
                                          model_config.eval):
    training_example.crop_supervision(model_config.hparams.clc)
  return training_example

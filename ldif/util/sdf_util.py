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
"""Utilties for working with sdf and pseudo-sdf functions."""

import tensorflow as tf


def apply_class_transfer(sdf, model_config, soft_transfer, offset, dtype=None):
  """Applies a class label transformation to an input sdf.

  Args:
    sdf: Tensor of any shape. The sdfs to transform elementwise.
    model_config: A ModelConfig object.
    soft_transfer: Boolean. Whether the input gt should have a smooth
      classification transfer applied, or a hard one. A hard transformation
      destroys gradients.
    offset: The isolevel at which the surface lives.
    dtype: tf.float32 or tf.bool if specified. The output type. A soft transfer
      is always a float32, so this parameter is ignored if soft_transfer is
      true. If soft_transfer is false, a cast from bool to float32 is made if
      necessary. Defaults to tf.float32.

  Returns:
    classes: Tensor of the same shape as sdf.
  """
  # If the prediction defines the surface boundary at a location other than
  # zero, we have to offset before we apply the classification transfer:
  if offset:
    sdf -= offset
  if soft_transfer:
    if model_config.hparams.lhdn == 't':
      with tf.variable_scope('lhdn', reuse=tf.AUTO_REUSE):
        tf.logging.info('Getting hdn variable in scope %s',
                        tf.get_variable_scope().name)
        init_value = tf.constant([model_config.hparams.hdn], dtype=tf.float32)
        hdn = tf.get_variable(
            'hdn', dtype=tf.float32, initializer=init_value, trainable=True)
    else:
      hdn = model_config.hparams.hdn
    return tf.sigmoid(hdn * sdf)
  else:
    if dtype is None or dtype == tf.float32:
      return tf.cast(sdf > 0.0, dtype=tf.float32)
    else:
      return sdf > 0.0


def apply_density_transfer(sdf):
  """Applies a density transfer function to an input sdf.

  The input sdf could either be from a prediction or the ground truth.

  Args:
    sdf: Tensor of any shape. The sdfs to transform elementwise.

  Returns:
    densities: Tensor of the same shape as sdf. Contains values in the range
    0-1, where it is 1 near the surface. The density is not a pdf, as it does
    not sum to 1 over the tensor.
  """
  # TODO(kgenova) This is one of the simplest possible transfer functions,
  # but is it the right one? Falloff rate should be controlled, and a 'signed'
  # density might even be relevant.
  return tf.exp(-tf.abs(sdf))

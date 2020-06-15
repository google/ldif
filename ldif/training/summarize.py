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
"""Summarizes the predictions."""

from matplotlib import cm
import numpy as np
import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order


def summarize_max_rbf_slices(rbfs, model_config):
  """Returns a set of slices of the RBFs.

  Args:
    rbfs: Tensor with shape [batch_size, quadric_count, length, height, width,
      1].
    model_config: A model configuration object.

  Returns:
  """
  batch_size, quadric_count, length, height, width = rbfs.get_shape().as_list(
  )[:5]
  palette = cm.get_cmap('plasma', quadric_count)
  color_rgbas = [palette(i) for i in range(quadric_count)]
  color_rgbas = np.stack(color_rgbas, axis=0)
  color_rgbs = color_rgbas[:, :3]
  color_rgbs = tf.constant(color_rgbs, dtype=tf.float32)
  dataset_split = model_config.inputs['split']

  five_dim_rbf = tf.reshape(rbfs,
                            [batch_size, quadric_count, length, height, width])
  max_rbf = tf.math.argmax(five_dim_rbf, axis=1)
  # max_rbf = tf.reshape(max_rbf, [batch_size, length, height, width, 1])
  visualization = tf.gather(color_rgbs, max_rbf)
  slices = []
  for li in range(length):
    if li % 12 == 2 or li == length // 2:
      slices.append(visualization[:, li, ...])

  slice_image = tf.concat(slices, axis=1)
  tf.summary.image(
      '%s-max_rbf_slices' % dataset_split, slice_image, max_outputs=4)

  mean_max_weight = tf.reduce_mean(tf.reduce_max(five_dim_rbf, axis=1))
  tf.summary.scalar('%s-mean_max_rbf_weight' % dataset_split, mean_max_weight)
  peak_weight = tf.reduce_max(five_dim_rbf)
  tf.summary.scalar('%s-peak_rbf_weight' % dataset_split, peak_weight)
  mean_total_weight = tf.reduce_mean(tf.reduce_sum(five_dim_rbf, axis=1))
  tf.summary.scalar('%s-mean_total_rbf_weight' % dataset_split,
                    mean_total_weight)

  frac_of_biggest = tf.divide(
      tf.reduce_max(five_dim_rbf, axis=1), tf.reduce_sum(five_dim_rbf, axis=1))
  mean_frac_of_biggest = tf.reduce_mean(frac_of_biggest)
  tf.summary.scalar('%s-mean_plurality_frac' % dataset_split,
                    mean_frac_of_biggest)

  pixel_count_above_half_weight = tf.count_nonzero(
      five_dim_rbf > 0.5, axis=[2, 3, 4]) / (
          length * height * width)
  tf.summary.histogram('%s-pixel_count_above_half_weight' % dataset_split,
                       pixel_count_above_half_weight)
  pixel_count_above_tenth_weight = tf.count_nonzero(
      five_dim_rbf > 0.1, axis=[2, 3, 4]) / (
          length * height * width)
  tf.summary.histogram('%s-pixel_count_above_tenth_weight' % dataset_split,
                       pixel_count_above_tenth_weight)


# TODO(kgenova) Make a summarizer class that handles all this so the model
# config doesn't need to get passed over and over, and so that it's easy to
# deal with the tensorflow boilerplate.
def summarize_in_out_image(model_config, prediction):  # pylint:disable=unused-argument
  log.info('In-out image summaries have been removed.')


def summarize_loss(model_config, loss, name):
  dataset_split = model_config.inputs['split']
  tf.summary.scalar('%s-loss/%s' % (dataset_split, name), loss)


def add_train_summaries(model_config, prediction):
  summarize_in_out_image(model_config, prediction)

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
"""Metrics for evaluating structured implicit functions."""

import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import sdf_util
# pylint: enable=g-bad-import-order


def point_iou(structured_implicit, sample_locations, sample_gt, model_config):
  """Estimates the mesh iou by taking the iou from uniform point samples."""
  assert model_config.hparams.bs == 1  # Otherwise the result would be wrong

  pred_class, _ = structured_implicit.class_at_samples(sample_locations)

  gt_is_inside = tf.logical_not(sdf_util.apply_class_transfer(
      sample_gt, model_config, soft_transfer=False, offset=0.0,
      dtype=tf.bool))
  pred_is_inside = pred_class < 0.5
  intersection = tf.cast(
      tf.logical_and(gt_is_inside, pred_is_inside), dtype=tf.float32)
  union = tf.cast(tf.logical_or(gt_is_inside, pred_is_inside), dtype=tf.float32)

  iou = tf.divide(tf.reduce_sum(intersection), tf.reduce_sum(union) + 1e-05)
  return iou

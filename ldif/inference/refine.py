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
"""Refines a prediction according to an observation using gradient descent."""

import numpy as np
import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.training import loss
# pylint: enable=g-bad-import-order


def sample_gradients_wrt_placeholders(model_config, training_example,
                                      prediction, samples):
  """Creates gradient ops for refining a quadric prediction."""
  structured_implicit_ph = prediction.structured_implicit.as_placeholder()
  sample_shape = samples.get_shape().as_list()
  target_shape = samples.get_shape().as_list()
  target_shape[-1] = 1  # The target is a function value, not an xyz position.
  samples_ph = tf.placeholder(tf.float32, shape=sample_shape)
  target_f_ph = tf.placeholder(tf.float32, shape=target_shape)

  # TODO(kgenova): One way to test whether trilinear interpolation is a problem
  # or it's some gradient sharing (?) is to replace this with an on-grid
  # evaluation.

  loss_val = loss.sample_loss(
      model_config,
      target_f_ph,
      structured_implicit_ph,
      samples_ph,
      name=None,
      apply_ucf=False)
  gradients = tf.gradients(loss_val, structured_implicit_ph.tensor_list)

  render = tf.zeros([model_config.hparams.bs, 256, 256, 3], dtype=tf.float32)

  render_alpha = tf.ones_like(render)
  tiled_render = tf.concat(3 * [render] + [render_alpha], axis=-1)
  batch_size = render.get_shape().as_list()[0]
  original_vis_ph = tf.placeholder(
      tf.float32, shape=[batch_size, 256, 84 + 256 * 6, 4])
  render_im = tf.concat([original_vis_ph, tiled_render], axis=2)
  # Keep this summary off the global collection so that it isn't evaluated
  # each batch (which would fail due to the placeholders).
  render_summary_op = tf.summary.image(
      '%s-iterative-refinement-result' % training_example.split,
      render_im,
      collections=[],
      max_outputs=32)

  return (structured_implicit_ph, samples_ph, target_f_ph, gradients,
          render_summary_op, original_vis_ph, tiled_render)


def refine(predicted_representation_ph, target_ph, samples_ph, original_vis_ph,
           gradient_op, predicted_representation_np, target_np, samples_np,
           original_vis_np, session, render_summary_op, iterated_render_tf):
  """Uses gradient descent to iteratively refine a prediction."""
  to_update = [np.copy(x) for x in predicted_representation_np]
  target_np = np.copy(target_np)
  samples_np = np.copy(samples_np)
  # (quadrics_np, centers_np, radii_np) = predicted_representation_np
  to_update_tf = predicted_representation_ph
  step_count = 200
  # to_update = [quadrics_np, centers_np, radii_np]
  for _ in range(step_count):
    feed_dict = {
        samples_ph: samples_np,
        target_ph: target_np,
    }
    for i in range(len(to_update)):
      feed_dict[to_update_tf[i]] = to_update[i]
    gradients_np = session.run(gradient_op, feed_dict=feed_dict)
    # log.info(gradients_np)
    # [1.5e-2, 7.5e-4, 5e-6]
    # TODO(kgenova): Try updating the gradients/output ranges to respect this
    # quadric_step_size = 1.5e-4
    # center_step_size = 7.5e-5
    # radius_step_size = 7.5e-7
    # step_sizes = [quadric_step_size, center_step_size, radius_step_size]
    step_sizes = 3 * [7.5e-5]
    for i in range(len(gradients_np)):
      # step_size = 2.5e-3  * 0.075#1e-3
      step_size = step_sizes[i]
      # if si >= 200:
      #   step_size /= 2.0
      # if si >= 400:
      #   step_size /= 2.0
      update = -step_size * gradients_np[i]
      to_update[i] += update
  # Render the result:
  feed_dict = {original_vis_ph: original_vis_np}
  for i in range(len(to_update)):
    feed_dict[to_update_tf[i]] = to_update[i]
  render_summary, iterated_render_np = session.run(
      [render_summary_op, iterated_render_tf], feed_dict=feed_dict)
  return render_summary, iterated_render_np

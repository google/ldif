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
"""Nets for predicting 2D lines."""

import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import line_util
from ldif.util import net_util
# pylint: enable=g-bad-import-order


def lstm_lines_transcoder(inputs, model_config):
  """A model that uses an LSTM to predict a sequence of lines."""
  batch_size = model_config.hparams.bs
  height = model_config.hparams.h
  width = model_config.hparams.w
  conv_layer_depths = [16, 32, 64, 128, 128]
  number_of_lines = model_config.hparams.sc  # Was 4!
  number_of_layers = model_config.hparams.cc
  line_embedding_size = 5
  encoded_image_length = 1024
  # embedding_size = line_embedding_size * number_of_lines
  lstm_size = model_config.hparams.hlw
  reencode_partial_outputs = True

  encoded_image, _, _ = net_util.encode(
      inputs,
      model_config,
      conv_layer_depths=conv_layer_depths,
      fc_layer_depths=[encoded_image_length],
      name='Encoder')
  if reencode_partial_outputs:
    encoded_partial_output, _, _ = net_util.encode(
        tf.ones_like(inputs),
        model_config,
        conv_layer_depths=conv_layer_depths,
        fc_layer_depths=[encoded_image_length],
        name='OutputReencoder')
  else:
    encoded_partial_output = None

  lstm = tf.contrib.rnn.MultiRNNCell([
      tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(number_of_layers)
  ])

  state = lstm.zero_state(batch_size, tf.float32)
  partial_images = []
  for i in range(number_of_lines):
    if reencode_partial_outputs:
      embedding_in = tf.concat([encoded_image, encoded_partial_output], axis=-1)
    else:
      embedding_in = encoded_image
    output, state = lstm(embedding_in, state)
    line_parameters = tf.contrib.layers.fully_connected(
        inputs=output,
        num_outputs=line_embedding_size,
        activation_fn=None,
        normalizer_fn=None)
    current_image = line_util.network_line_parameters_to_line(
        line_parameters, height, width)
    current_image = 2.0 * current_image - 1.0
    if partial_images:
      current_image = tf.minimum(partial_images[-1], current_image)
    partial_images.append(current_image)

    if reencode_partial_outputs and i < number_of_lines - 1:
      encoded_partial_output, _, _ = net_util.encode(
          current_image,
          model_config,
          conv_layer_depths=conv_layer_depths,
          fc_layer_depths=[encoded_image_length],
          name='OutputReencoder')
  return partial_images


def bag_of_lines_transcoder(inputs, model_config):
  """A model that outputs K lines all at once to replicate a target shape."""
  batch_size = model_config.hparams.bs
  height = model_config.hparams.h
  width = model_config.hparams.w
  conv_layer_depths = [16, 32, 64, 128, 128]
  number_of_lines = 4
  line_embedding_size = 5
  embedding_size = line_embedding_size * number_of_lines
  z, _, _ = net_util.encode(
      inputs,
      model_config,
      conv_layer_depths=conv_layer_depths,
      fc_layer_depths=[embedding_size],
      name='Encoder')
  z = tf.reshape(z, [batch_size, number_of_lines, line_embedding_size])

  # Code is not batched:
  batch_items = []
  with tf.name_scope('Decoder'):
    for bi in range(batch_size):
      line_images = []
      for i in range(number_of_lines):
        line_parameters = z[bi, i, :]
        line_image = line_util.network_line_parameters_to_line(
            line_parameters, height, width)
        line_images.append(line_image)
      batch_items.append(line_util.union_of_line_drawings(line_images))
    decoded_batch = tf.stack(batch_items, axis=0)
    # Lines are in the range [0,1], so rescale to [-1, 1].
    return 2.0 * decoded_batch - 1.0

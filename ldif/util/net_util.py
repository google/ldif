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
"""Utilities for building neural networks."""

import functools
import tensorflow as tf
import tensorflow.contrib.layers as contrib_layers

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order


def twin_inference(inference_fun, observation, element_count,
                   flat_element_length, model_config):
  """Predicts both explicit and implicit parameters for each shape element."""
  element_embedding_length = model_config.hparams.ips
  # if element_embedding_length == 0:
  # raise ValueError(
  # 'Twin Inference networks requires a nonzero imlicit parameter count.')
  remaining_length = flat_element_length - element_embedding_length
  if remaining_length <= 0:
    log.warning('Using less-tested option: single-tower in twin-tower.')
    explicit_embedding_length = flat_element_length
  else:
    explicit_embedding_length = remaining_length
  if model_config.hparams.ips <= 10:
    raise ValueError('Unsafe code: May not be possible to determine '
                     'presence/absence of implicit parameters')

  # TODO(kgenova) These scope names aren't great but we don't want to mess up
  # old checkpoints. Once old checkpoints are irrelevant, improve them.
  with tf.variable_scope('explicit_embedding_cnn'):
    prediction, embedding = inference_fun(observation, element_count,
                                          explicit_embedding_length,
                                          model_config)
  if remaining_length > 0:
    scope_name = 'implicit_embedding_cnn' + (
        '_v2' if model_config.hparams.iec == 'v2' else '')
    with tf.variable_scope(scope_name):
      implicit_parameters, implicit_embedding = inference_fun(
          observation, element_count, element_embedding_length, model_config)
      prediction = tf.concat([prediction, implicit_parameters], axis=2)
      embedding = tf.concat([embedding, implicit_embedding], axis=1)

  return prediction, embedding


def get_normalizer_and_mode(model_config):
  """Sets up the batch normalizer mode."""
  if model_config.hparams.nrm == 'bn':
    normalizer_params = {'is_training': model_config.train, 'trainable': True}
    normalizer = contrib_layers.batch_norm
  elif model_config.hparams.nrm == 'bneit':
    normalizer_params = {
        'is_training': not model_config.inference,
        'trainable': True
    }
    normalizer = contrib_layers.batch_norm
  elif model_config.hparams.nrm == 'bneitvt':
    normalizer_params = {
        'is_training': not model_config.inference,
        'trainable': not model_config.inference
    }
    normalizer = contrib_layers.batch_norm
  elif model_config.hparams.nrm == 'none':
    normalizer_params = None
    normalizer = None
  else:
    raise ValueError('The input normalization hyperparameter %s is unknown.' %
                     model_config.hparams.nrm)
  return normalizer, normalizer_params


def conv_layer(inputs, depth, model_config):
  """A single 3x3 convolutional layer with stride 1."""
  normalizer, normalizer_params = get_normalizer_and_mode(model_config)
  ibs, ih, iw, ic = inputs.get_shape().as_list()
  outputs = contrib_layers.convolution(
      inputs=inputs,
      num_outputs=depth,
      kernel_size=3,
      stride=1,
      padding='SAME',
      normalizer_fn=normalizer,
      normalizer_params=normalizer_params,
      activation_fn=tf.nn.leaky_relu)
  obs, oh, ow, oc = outputs.get_shape().as_list()
  log.info(
      'Applying conv+relu layer. Input: [%i, %i, %i, %i]. Output: [%i, %i, %i, %i].'
      % (ibs, ih, iw, ic, obs, oh, ow, oc))
  return outputs


def maxpool2x2_layer(inputs):
  """A maxpool layer that pools 2x2 -> 1."""
  assert len(inputs.get_shape().as_list()) == 4
  batch_size_in, height_in, width_in, channel_count_in = inputs.get_shape(
  ).as_list()
  outputs = tf.layers.max_pooling2d(
      inputs,
      pool_size=2,
      strides=2,
      padding='same',
      data_format='channels_last',
      name=None)

  # outputs = tf.nn.max_pool(
  #     inputs, ksize=[0, 2, 2, 0], strides=[1, 2, 2, 1], padding='SAME')
  batch_size_out, height_out, width_out, channel_count_out = outputs.get_shape(
  ).as_list()
  assert batch_size_in == batch_size_out
  assert height_out * 2 == height_in
  assert width_out * 2 == width_in
  assert channel_count_in == channel_count_out
  log.info(
      'Applying maxpool2x2 layer. Input: [%i, %i, %i, %i]. Output: [%i, %i, %i, %i].'
      % (batch_size_in, height_in, width_in, channel_count_in, batch_size_out,
         height_out, width_out, channel_count_out))
  return outputs


def down_layer(inputs, depth, model_config):
  """A single encoder layer."""
  normalizer, normalizer_params = get_normalizer_and_mode(model_config)
  return contrib_layers.convolution(
      inputs=inputs,
      num_outputs=depth,
      kernel_size=3,
      stride=2,
      padding='SAME',
      normalizer_fn=normalizer,
      normalizer_params=normalizer_params,
      activation_fn=tf.nn.leaky_relu)


def up_layer(inputs, spatial_dims, depth, model_config):
  """A single decoder layer."""
  normalizer, normalizer_params = get_normalizer_and_mode(model_config)
  if len(tf.shape(inputs)) != 4:
    if len(tf.shape(inputs)) != 5:
      raise ValueError('Unexpected input dimensionality: %i' %
                       len(tf.shape(inputs)))
    raise ValueError('3D Upsampling has not been implemented.')
  layer = contrib_layers.convolution(
      inputs=inputs,
      num_outputs=depth,
      kernel_size=5,
      padding='SAME',
      normalizer_fn=normalizer,
      normalizer_params=normalizer_params,
      activation_fn=tf.nn.leaky_relu)
  return tf.image.resize_images(
      images=layer, size=spatial_dims, align_corners=True)


def encode(inputs,
           model_config,
           conv_layer_depths,
           fc_layer_depths,
           name='Encoder'):
  """Encodes an input image batch as a latent vector."""
  # This encoder first applies 3x3 stride 2 conv layers with the given feature
  # depths. Then, the vector is reshaped and the fully connected layers are
  # applied with the given depths.
  with tf.variable_scope(name):
    normalizer, normalizer_params = get_normalizer_and_mode(model_config)
    batch_size = model_config.hparams.bs
    net = inputs
    endpoints = {}
    input_shape = inputs.get_shape().as_list()
    # The first dimension is the batch dimension; the last is the channel dim.
    # All other dimensions should be spatial.
    spatial_dims = [tuple(input_shape[1:-1])]

    for i, depth in enumerate(conv_layer_depths):
      net = down_layer(net, depth, model_config)
      endpoints['encoder_%d' % i] = net
      net_shape = net.get_shape().as_list()
      spatial_dims.append(tuple(net_shape[1:-1]))

    current_total_dimensionality = functools.reduce(
        lambda x, y: x * y, spatial_dims[-1]) * conv_layer_depths[-1]

    net = tf.reshape(net, [batch_size, current_total_dimensionality])

    for depth in fc_layer_depths:
      net = contrib_layers.fully_connected(
          inputs=net,
          num_outputs=depth,
          activation_fn=tf.nn.leaky_relu,
          normalizer_fn=normalizer,
          normalizer_params=normalizer_params)
  return net  #, spatial_dims, endpoints


def inputs_to_feature_vector(inputs, feature_length, model_config):
  """Encodes an input observation tensor to a fixed lengthh feature vector."""
  batch_size, image_count, height, width, channel_count = (
      inputs.get_shape().as_list())
  log.verbose('Input shape to early-fusion cnn: %s' %
              str(inputs.get_shape().as_list()))
  if image_count == 1:
    im = tf.reshape(inputs, [batch_size, height, width, channel_count])
  else:
    im = tf.reshape(
        tf.transpose(inputs, perm=[0, 2, 3, 1, 4]),
        [batch_size, height, width, image_count * channel_count])
  embedding = encode(
      im,
      model_config,
      conv_layer_depths=[16, 32, 64, 128, 128],
      fc_layer_depths=[feature_length])
  return embedding


def decode(z,
           model_config,
           spatial_dims,
           fc_layer_depths,
           conv_layer_depths,
           output_dim,
           name='Decoder'):
  """Decode a latent vector into an image."""
  # z: Latent vector with shape [batch_size, latent_dimensionality].
  with tf.variable_scope(name):
    normalizer, normalizer_params = get_normalizer_and_mode(model_config)
    batch_size = model_config.hparams.bs
    net = z
    for depth in fc_layer_depths:
      net = contrib_layers.fully_connected(
          inputs=net,
          num_outputs=depth,
          activation_fn=tf.nn.leaky_relu,
          normalizer_fn=normalizer,
          normalizer_params=normalizer_params)

    # We assume that the output dimensionality chosen in the last fc layer
    # is appropriate to reshape and scale.
    fc_width = model_config.hparams.w // 2**(len(conv_layer_depths))
    fc_height = model_config.hparams.h // 2**(len(conv_layer_depths))

    net = tf.reshape(net,
                     [batch_size, fc_height, fc_width, conv_layer_depths[-1]])

    for i, depth in enumerate(reversed(conv_layer_depths)):
      net = up_layer(net, spatial_dims[-(i + 2)], depth, model_config)

    generated = contrib_layers.conv2d(
        inputs=net,
        num_outputs=output_dim,
        kernel_size=1,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.tanh)
  return generated, None


def residual_layer(inputs, num_outputs, model_config):
  """A single residual network layer unit."""
  normalizer, normalizer_params = get_normalizer_and_mode(model_config)
  if normalizer is not None:
    output = normalizer(
        inputs,
        is_training=normalizer_params['is_training'],
        trainable=normalizer_params['trainable'])
  else:
    output = inputs
  output = tf.nn.leaky_relu(output)
  output = contrib_layers.fully_connected(
      inputs=output,
      num_outputs=num_outputs,
      activation_fn=None,
      normalizer_fn=None)
  if normalizer is not None:
    output = normalizer(
        output,
        is_training=normalizer_params['is_training'],
        trainable=normalizer_params['trainable'])
  output = tf.nn.leaky_relu(output)
  output = contrib_layers.fully_connected(
      inputs=output,
      num_outputs=num_outputs,
      activation_fn=None,
      normalizer_fn=None)
  return output + inputs

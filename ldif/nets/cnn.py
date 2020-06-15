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
"""A simple feed forward CNN."""

import functools

import tensorflow as tf
import tensorflow.contrib.layers as contrib_layers
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.nets import resnet_v2 as contrib_slim_resnet_v2
import tensorflow_hub as hub

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import net_util
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order


def early_fusion_cnn(observation, element_count, element_length, model_config):
  """A cnn that maps 1+ images with 1+ chanels to a feature vector."""
  inputs = observation.tensor
  if model_config.hparams.cnna == 'cnn':
    embedding = net_util.inputs_to_feature_vector(inputs, 1024, model_config)
  elif model_config.hparams.cnna in ['r18', 'r50', 'h50', 'k50', 's50']:
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
    if model_config.hparams.cnna == 'r18':
      raise ValueError('ResNet18 is no longer supported.')
    elif model_config.hparams.cnna == 'r50':
      raise ValueError('r50 network is no longer supported.')
    elif model_config.hparams.cnna == 'k50':
      log.warning('Using a keras based model.')
      resnet = tf.compat.v1.keras.applications.ResNet50V2(
          include_top=False,
          weights=None,
          input_tensor=None,
          input_shape=(height, width, image_count * channel_count),
          pooling=None)
      embedding = resnet(im)
    elif model_config.hparams.cnna == 's50':
      with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        embedding_prenorm, _ = contrib_slim_resnet_v2.resnet_v2_50(
            inputs=im,
            num_classes=2048,
            is_training=model_config.train,
            global_pool=True,
            reuse=tf.AUTO_REUSE,
            scope='resnet_v2_50')
        embedding_prenorm = tf.reshape(embedding_prenorm,
                                       [model_config.hparams.bs, 2048])
        embedding = tf.nn.l2_normalize(embedding_prenorm, axis=1)
    elif model_config.hparams.cnna == 'h50':
      log.warning('TF Hub not tested externally.')
      resnet = hub.Module(
          'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1',
          trainable=True)
      expected_height, expected_width = hub.get_expected_image_size(resnet)
      if channel_count == 1:
        im = tf.tile(im, [1, 1, 1, 3])
      if height != expected_height or width != expected_width:
        raise ValueError(
            ('The input tensor has shape %s, but this tf.Hub()'
             ' r50 expects [%i, %i, 3].') %
            (repr(im.get_shape().as_list()), expected_height, expected_width))
      embedding = resnet(im)
    log.verbose('Embedding shape: %s' % repr(embedding.get_shape().as_list()))
    current_total_dimensionality = functools.reduce(
        lambda x, y: x * y,
        embedding.get_shape().as_list()[1:])
    embedding = tf.reshape(
        embedding, [model_config.hparams.bs, current_total_dimensionality])

  net = embedding
  normalizer, normalizer_params = net_util.get_normalizer_and_mode(model_config)
  for _ in range(2):
    net = contrib_layers.fully_connected(
        inputs=net,
        num_outputs=2048,
        activation_fn=tf.nn.leaky_relu,
        normalizer_fn=normalizer,
        normalizer_params=normalizer_params)
  prediction = contrib_layers.fully_connected(
      inputs=net,
      num_outputs=element_count * element_length,
      activation_fn=None,
      normalizer_fn=None)
  batch_size = inputs.get_shape().as_list()[0]
  prediction = tf.reshape(prediction,
                          [batch_size, element_count, element_length])
  return prediction, embedding


def mid_fusion_cnn(observation, element_count, element_length, model_config):
  """A CNN architecture that fuses individual image channels in the middle."""
  with tf.name_scope('mid_fusion_cnn'):
    inputs = observation.tensor

    individual_images = tf.split(
        inputs, num_or_size_splits=inputs.get_shape().as_list()[1], axis=1)
    image_count = len(individual_images)
    assert image_count == model_config.hparams.rc  # just debugging, can remove.
    embeddings = []
    embedding_length = 1023
    for i, image in enumerate(individual_images):
      with tf.variable_scope('mfcnn', reuse=i != 0):
        embedding = net_util.inputs_to_feature_vector(image, embedding_length,
                                                      model_config)
        if model_config.hparams.fua == 't':
          embedding = tf.reshape(
              embedding, [model_config.hparams.bs, 3, embedding_length // 3])
          embedding = tf.pad(embedding, tf.constant([[0, 0], [0, 1], [0, 0]]))
          cam2world_i = observation.cam_to_worlds[:, i, :, :]  # [bs, rc, 4, 4]
          embedding = tf.matmul(cam2world_i, embedding)
          # embedding shape [bs, 4, embedding_length // 3]
          embedding = tf.reshape(embedding[:, :3, :],
                                 [model_config.hparams.bs, embedding_length])
        embeddings.append(embedding)
    if image_count > 1:
      embeddings = tf.ensure_shape(
          tf.stack(embeddings, axis=1),
          [model_config.hparams.bs, image_count, embedding_length])
      embedding = tf.reduce_max(embeddings, axis=1)
    else:
      embedding = embeddings[0]
    # TODO(kgenova) Should be a helper:
    net = embedding
    normalizer, normalizer_params = net_util.get_normalizer_and_mode(
        model_config)
    for _ in range(2):
      net = contrib_layers.fully_connected(
          inputs=net,
          num_outputs=2048,
          activation_fn=tf.nn.leaky_relu,
          normalizer_fn=normalizer,
          normalizer_params=normalizer_params)
    prediction = contrib_layers.fully_connected(
        inputs=net,
        num_outputs=element_count * element_length,
        activation_fn=None,
        normalizer_fn=None)
    batch_size = inputs.get_shape().as_list()[0]
    prediction = tf.reshape(prediction,
                            [batch_size, element_count, element_length])
  return prediction, embedding

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
"""Implementation of PointNet network."""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as contrib_layers

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import math_util
# pylint: enable=g-bad-import-order


def point_set_to_transformation(points):
  """Maps a point set to an affine transformation and a translation."""
  batch_size, point_count, _ = points.get_shape().as_list()
  with tf.variable_scope('r3_transformation_net'):
    net = tf.expand_dims(points, axis=2)
    net = contrib_layers.conv2d(
        inputs=net,
        num_outputs=64,
        kernel_size=[1, 1],
        padding='VALID',
        stride=[1, 1],
        scope='conv1')
    net = contrib_layers.conv2d(
        net,
        num_outputs=128,
        kernel_size=[1, 1],
        padding='VALID',
        stride=[1, 1],
        scope='conv2')
    net = contrib_layers.conv2d(
        net,
        num_outputs=1024,
        kernel_size=[1, 1],
        padding='VALID',
        stride=[1, 1],
        scope='conv3')
    net = contrib_layers.max_pool2d(
        net, kernel_size=[point_count, 1], padding='VALID', scope='maxpool1')
    net = contrib_layers.flatten(net)
    net = contrib_layers.fully_connected(
        net, num_outputs=512, activation_fn=tf.nn.relu, scope='fc1')
    net = contrib_layers.fully_connected(net, num_outputs=256, scope='fc2')

    with tf.variable_scope('transformation'):
      weights = tf.get_variable(
          'weights', [256, 3 * 3],
          initializer=tf.constant_initializer(0.0),
          dtype=tf.float32)
      biases = tf.get_variable(
          'biases', [3 * 3],
          initializer=tf.constant_initializer(0.0),
          dtype=tf.float32)
      biases += tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32)
      transformation = tf.matmul(net, weights)
      transformation = tf.nn.bias_add(transformation, biases)
      transformation = tf.reshape(transformation, [batch_size, 3, 3])
    with tf.variable_scope('translation'):
      translation_weights = tf.get_variable(
          'weights', [256, 3],
          initializer=tf.constant_initializer(0.0),
          dtype=tf.float32)
      translation = tf.matmul(net, translation_weights)
      translation = tf.reshape(translation, [batch_size, 1, 3])
    return transformation, translation


def point_set_to_feature_transformation(points, output_dimensionality):
  """A block to learn an orthogonal feature transformation matrix."""
  batch_size, point_count, _, input_feature_count = points.get_shape().as_list()
  assert input_feature_count == 64
  with tf.variable_scope('feature_transformation_net'):
    net = contrib_layers.conv2d(
        points,
        num_outputs=64,
        kernel_size=[1, 1],
        padding='VALID',
        stride=[1, 1],
        scope='conv1')
    net = contrib_layers.conv2d(
        net,
        num_outputs=128,
        kernel_size=[1, 1],
        padding='VALID',
        stride=[1, 1],
        scope='conv2')
    net = contrib_layers.conv2d(
        net,
        num_outputs=1024,
        kernel_size=[1, 1],
        padding='VALID',
        stride=[1, 1],
        scope='conv3')
    net = contrib_layers.max_pool2d(
        net, kernel_size=[point_count, 1], padding='VALID', scope='maxpool1')
    net = contrib_layers.flatten(net)
    net = contrib_layers.fully_connected(
        net, num_outputs=512, activation_fn=tf.nn.relu, scope='fc1')
    net = contrib_layers.fully_connected(net, num_outputs=256, scope='fc2')
    with tf.variable_scope('feature_transformation'):
      weights = tf.get_variable(
          'weights', [256, output_dimensionality * output_dimensionality],
          initializer=tf.constant_initializer(0.0),
          dtype=tf.float32)
      biases = tf.get_variable(
          'biases', [output_dimensionality * output_dimensionality],
          initializer=tf.constant_initializer(0.0),
          dtype=tf.float32)
      biases += tf.constant(
          np.eye(output_dimensionality).flatten(), dtype=tf.float32)
      transformation = tf.matmul(net, weights)
      transformation = tf.nn.bias_add(transformation, biases)
    transformation = tf.reshape(
        transformation,
        [batch_size, output_dimensionality, output_dimensionality])
    return transformation


def pointnet_depr(points,
                  output_feature_count,
                  apply_learned_ortho_tx=False,
                  apply_learned_64d_tx=True,
                  use_bad_reduce=False,
                  nerfify=False,
                  maxpool_feature_count=1024):
  """Applies pointnet to an input set of point features.

  Args:
    points: Tensor with shape [batch_size, point_count, feature_count].
    output_feature_count: The number of features in the final linear layer.
    apply_learned_ortho_tx: Whether to apply the learned transformation to the
      input points.
    apply_learned_64d_tx: Whether to apply the 64x64 learned orthogonal
      transform.
    use_bad_reduce: Whether to use the original slow 'maxpool2d' global
      max reduce. Only still an option for compatibility with existing trained
      networks.
    nerfify: Whether to apply the math_util.nerfify function to the features
      (all of them, not just the points) after the initial transform step.
    maxpool_feature_count: Integer. The number of features in the vector before
      doing a global maxpool. This is the main computational bottleneck, so
      reducing it is good for training time.

  Returns:
    embedding: Tensor with shape [batch_size, embedding_length].
  """
  batch_size, point_count, feature_count = points.get_shape().as_list()

  point_positions = points[..., 0:3]
  point_features = points[..., 3:]
  feature_count = points.get_shape().as_list()[-1] - 3
  with tf.variable_scope('pointnet', reuse=tf.AUTO_REUSE):
    if apply_learned_ortho_tx:
      with tf.variable_scope('learned_transformation'):
        transformation, translation = point_set_to_transformation(points)
        transformed_points = tf.matmul(point_positions + translation,
                                       transformation)
        if feature_count > 0:
          transformed_points = tf.concat([transformed_points, point_features],
                                         axis=2)
      net = tf.expand_dims(transformed_points, axis=2)
    else:
      net = tf.expand_dims(points, axis=2)

    if nerfify:
      net = math_util.nerfify(net, 10, flatten=True, interleave=False)

    # Apply the 'mlp 64, 64' layers:
    with tf.variable_scope('mlp_block_1'):
      net = contrib_layers.conv2d(
          net,
          num_outputs=64,
          kernel_size=[1, 1],
          padding='VALID',
          stride=[1, 1],
          scope='conv1')
      net = contrib_layers.conv2d(
          net,
          num_outputs=64,
          kernel_size=[1, 1],
          padding='VALID',
          stride=[1, 1],
          scope='conv2')

    if apply_learned_64d_tx:
      with tf.variable_scope('learned_feature_transformation'):
        feature_transformation = point_set_to_feature_transformation(
            net, output_dimensionality=64)
        net = tf.matmul(
            tf.reshape(net, [batch_size, point_count, 64]),
            feature_transformation)
        net = tf.expand_dims(net, axis=2)

    # Second MLP block
    with tf.variable_scope('mlp_block_2'):
      net = contrib_layers.conv2d(
          net,
          num_outputs=64,
          kernel_size=[1, 1],
          padding='VALID',
          stride=[1, 1],
          scope='conv1')
      net = contrib_layers.conv2d(
          net,
          num_outputs=128,
          kernel_size=[1, 1],
          padding='VALID',
          stride=[1, 1],
          scope='conv2')
      net = contrib_layers.conv2d(
          net,
          num_outputs=maxpool_feature_count,  # TODO(kgenova) A bottleneck.
          kernel_size=[1, 1],
          padding='VALID',
          stride=[1, 1],
          scope='conv3')

    # log.info(f'Hello in pointnet. The shape is {net.get_shape().as_list()}')
    # raise ValueError('Stop')
    assert len(net.get_shape().as_list()) == 4
    if use_bad_reduce:
      net = contrib_layers.max_pool2d(
          net, [point_count, 1],
          stride=[2, 2],
          padding='VALID',
          scope='global_maxpool')
    else:
      net = tf.reshape(net, [batch_size, point_count, maxpool_feature_count])
      net = tf.reduce_max(net, axis=1)

    net = contrib_layers.flatten(net)

    # Final MLP
    with tf.variable_scope('final_mlp'):
      net = contrib_layers.fully_connected(
          net, num_outputs=512, activation_fn=tf.nn.relu, scope='fc1')
      net = contrib_layers.fully_connected(
          net, num_outputs=256, activation_fn=tf.nn.relu, scope='fc2')
      net = contrib_layers.fully_connected(
          net,
          num_outputs=output_feature_count,
          activation_fn=None,
          scope='final_fc')
    return net


def pointnet(points,
             output_feature_count,
             apply_learned_ortho_tx=False,
             apply_learned_64d_tx=True,
             use_bad_reduce=False,
             nerfify=False,
             maxpool_feature_count=1024,
             use_gpu=True):
  """Applies pointnet to an input set of point features.

  Args:
    points: Tensor with shape [batch_size, point_count, feature_count].
    output_feature_count: The number of features in the final linear layer.
    apply_learned_ortho_tx: Whether to apply the learned transformation to the
      input points.
    apply_learned_64d_tx: Whether to apply the 64x64 learned orthogonal
      transform.
    use_bad_reduce: Whether to use the original slow 'maxpool2d' global
      max reduce. Only still an option for compatibility with existing trained
      networks.
    nerfify: Whether to apply the math_util.nerfify function to the features
      (all of them, not just the points) after the initial transform step.
    maxpool_feature_count: Integer. The number of features in the vector before
      doing a global maxpool. This is the main computational bottleneck, so
      reducing it is good for training time.
    use_gpu: Whether to assume a GPU is available.

  Returns:
    embedding: Tensor with shape [batch_size, embedding_length].
  """
  batch_size, point_count, feature_count = points.get_shape().as_list()

  point_positions = points[..., 0:3]
  point_features = points[..., 3:]
  feature_count = points.get_shape().as_list()[-1] - 3
  with tf.variable_scope('pointnet', reuse=tf.AUTO_REUSE):
    if apply_learned_ortho_tx:
      with tf.variable_scope('learned_transformation'):
        transformation, translation = point_set_to_transformation(points)
        transformed_points = tf.matmul(point_positions + translation,
                                       transformation)
        if feature_count > 0:
          transformed_points = tf.concat([transformed_points, point_features],
                                         axis=2)
          points = transformed_points
    # Go from NWC to NCW so that the final reduce can be faster.
    assert len(points.shape) == 3
    net = points
    if nerfify:
      net = math_util.nerfify(net, 10, flatten=True, interleave=False)
    # On the GPU HCW is substantially faster, but there is so NCW CPU kernel.
    # So in CPU mode we have to do NWC convolutions.
    if use_gpu:
      net = tf.transpose(net, perm=[0, 2, 1])
      data_format = 'NCW'
      reduce_dim = 2
    else:
      data_format = 'NWC'
      reduce_dim = 1

    # Apply the 'mlp 64, 64' layers:
    with tf.variable_scope('mlp_block_1'):
      # with tf.variable_scope('test_keras'):
      #   net = tf.keras.layers.Conv1D(filters=64,
      #                              kernel_size=1,
      #                              strides=1,
      #                              padding='valid',
      #                              data_format='channels_first',
      #                              activation=tf.keras.activations.relu)(net)

      net = contrib_layers.conv1d(
          net,
          num_outputs=64, kernel_size=1,
          padding='VALID',
          stride=1,
          data_format=data_format,
          scope='conv1')
      net = contrib_layers.conv1d(
          net,
          num_outputs=64,
          kernel_size=1,
          padding='VALID',
          stride=1,
          data_format=data_format,
          scope='conv2')

    if apply_learned_64d_tx:
      if use_gpu:
        net = tf.transpose(net, perm=[0, 2, 1])
      with tf.variable_scope('learned_feature_transformation'):
        feature_transformation = point_set_to_feature_transformation(
            net, output_dimensionality=64)
        net = tf.matmul(
            tf.reshape(net, [batch_size, point_count, 64]),
            feature_transformation)
        net = tf.expand_dims(net, axis=2)
      if use_gpu:
        net = tf.transpose(net, perm=[0, 2, 1])

    # Second MLP block
    with tf.variable_scope('mlp_block_2'):
      net = contrib_layers.conv1d(
          net,
          num_outputs=64,
          kernel_size=1,
          padding='VALID',
          stride=1,
          data_format=data_format,
          scope='conv1')
      net = contrib_layers.conv1d(
          net,
          num_outputs=128,
          kernel_size=1,
          padding='VALID',
          stride=1,
          data_format=data_format,
          scope='conv2')
      net = contrib_layers.conv1d(
          net,
          num_outputs=maxpool_feature_count,  # TODO(kgenova) A bottleneck.
          kernel_size=1,
          padding='VALID',
          stride=1,
          data_format=data_format,
          scope='conv3')

    # log.info(f'Hello in pointnet. The shape is {net.get_shape().as_list()}')
    # raise ValueError('Stop')
    assert len(net.get_shape().as_list()) == 3
    if use_bad_reduce:
      raise ValueError('Bad Reduce is not supported with pointnet1d.')

    net = tf.reduce_max(net, axis=reduce_dim)

    # net = contrib_layers.flatten(net)

    # Final MLP
    with tf.variable_scope('final_mlp'):
      net = contrib_layers.fully_connected(
          net, num_outputs=512, activation_fn=tf.nn.relu, scope='fc1')
      net = contrib_layers.fully_connected(
          net, num_outputs=256, activation_fn=tf.nn.relu, scope='fc2')
      net = contrib_layers.fully_connected(
          net,
          num_outputs=output_feature_count,
          activation_fn=None,
          scope='final_fc')
    return net

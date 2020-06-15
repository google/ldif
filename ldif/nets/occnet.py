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
"""An implementation of the OccNet architecture."""

import tensorflow as tf

from tensorflow.contrib import layers

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import net_util
from ldif.util import math_util
# pylint: enable=g-bad-import-order

# The epsilon used to pad before taking tf.sqrt(). Set equal to OccNet code.
SQRT_EPS = 1e-5


def dim_needs_broadcasting(a, b):
  return a != b and (a == 1 or b == 1)


def subset(shape, dims):
  """Returns the dims-th elements of shape."""
  out = []
  for dim in dims:
    out.append(shape[dim])
  return out


def shapes_equal(a, b, dims=None):
  if dims is None:
    dims = list(range(len(a)))
  a_shape = subset(a.get_shape().as_list(), dims)
  b_shape = subset(b.get_shape().as_list(), dims)
  for sa, sb in zip(a_shape, b_shape):
    if sa != sb:
      return False
  return True


def broadcast_if_necessary(a, b, dims):
  """Tiles shapes as necessary to match along a list of dims."""
  needs_broadcasting = False
  a_shape = a.get_shape().as_list()
  b_shape = b.get_shape().as_list()
  a_final_shape = []
  b_final_shape = []
  assert len(a_shape) == len(b_shape)
  for dim in range(len(a_shape)):
    if dim in dims and dim_needs_broadcasting(a_shape[dim], b_shape[dim]):
      needs_broadcasting = True
      dim_len = max(a_shape[dim], b_shape[dim])
      a_final_shape.append(dim_len)
      b_final_shape.append(dim_len)
    else:
      a_final_shape.append(a_shape[dim])
      b_final_shape.append(b_shape[dim])
  for dim in dims:
    if dim_needs_broadcasting(a_shape[dim], b_shape[dim]):
      needs_broadcasting = True
  if not needs_broadcasting:
    return a, b
  if not shapes_equal(a_shape, a_final_shape):
    a = tf.broadcast_to(a, a_final_shape)
  if not shapes_equal(b_shape, b_final_shape):
    b = tf.broadcast_to(b, b_final_shape)
  return a, b


def unbatched_fc_layer(vector, output_length, name):
  """Applies a linear fully connected layer to a single vector.

  Args:
    vector: Tensor with rank 1. The input vector.
    output_length: The number of activations in the output.
    name: String name for the local variable scope.

  Returns:
    Tensor with shape [output_length].
  """
  with tf.variable_scope(name):
    input_length = vector.get_shape().as_list()[0]
    vector = tf.reshape(vector, [1, input_length])
    output = layers.fully_connected(
        inputs=vector,
        num_outputs=output_length,
        activation_fn=None,
        normalizer_fn=None,
        normalizer_params=None)
  return tf.reshape(output, [output_length])


def fc_layer(vector, output_length, name):
  """Applies a linear fully connected layer to a single vector.

  Args:
    vector: Tensor with rank 1. The input vector.
    output_length: The number of activations in the output.
    name: String name for the local variable scope.

  Returns:
    Tensor with shape [output_length].
  """
  with tf.variable_scope(name):
    output = layers.fully_connected(
        inputs=vector,
        num_outputs=output_length,
        activation_fn=None,
        normalizer_fn=None,
        normalizer_params=None)
  return output


def ensure_is_scalar(t):
  s = t.get_shape().as_list()
  is_scalar = not s or (len(s) == 1 and s[0] == 1)
  if not is_scalar:
    raise ValueError('Expected tensor with shape %s to be a scalar tensor.' % s)


def batched_cbn_layer(shape_embedding, sample_embeddings, name, model_config):
  """Applies conditional batch norm to a batch of sample embeddings.

  The batch norm values are conditioned on shape embedding.

  Args:
    shape_embedding: Tensor with shape [batch_size, shape_embedding_length].
    sample_embeddings: Tensor with shape [batch_size, sample_count,
      sample_embedding_length].
    name: String naming the layer.
    model_config: A ModelConfig object.

  Returns:
    Tensor with shape [shape_embedding_length].
  """
  with tf.variable_scope(name):
    batch_size = shape_embedding.get_shape().as_list()[0]
    sample_embedding_length = sample_embeddings.get_shape().as_list()[2]
    beta = fc_layer(shape_embedding, sample_embedding_length, 'beta_fc')
    gamma = fc_layer(shape_embedding, sample_embedding_length, 'gamma_fc')
    batch_mean, batch_variance = tf.nn.moments(sample_embeddings, axes=[1, 2])
    ensure_shape(batch_mean, [batch_size])
    reduced_batch_mean = tf.reduce_mean(batch_mean)
    ensure_shape(batch_variance, [batch_size])
    reduced_batch_variance = tf.reduce_mean(batch_variance)
    running_mean = tf.get_variable(
        'running_mean',
        shape=[1],
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=False)
    running_variance = tf.get_variable(
        'running_variance',
        shape=[1],
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=False)
    is_training = model_config.train
    if is_training:
      running_mean = tf.assign(
          running_mean, 0.995 * running_mean + 0.005 * reduced_batch_mean)
      running_variance = tf.assign(
          running_variance,
          0.995 * running_variance + 0.005 * reduced_batch_variance)
      with tf.control_dependencies([running_mean, running_variance]):
        mean = tf.identity(running_mean)
        variance = tf.identity(running_variance)
    else:
      mean = running_mean
      variance = running_variance

    denom = tf.sqrt(variance + SQRT_EPS)
    out = (
        tf.expand_dims(gamma, axis=1) * tf.divide(
            (sample_embeddings - mean), denom) + tf.expand_dims(beta, axis=1))
    return out


def cbn_layer(shape_embedding, sample_embeddings, name, model_config):
  """Applies conditional batch norm to a batch of sample embeddings.

  The batch norm values are conditioned on shape embedding.

  Args:
    shape_embedding: Tensor with shape [shape_embedding_length].
    sample_embeddings: Tensor with shape [sample_count,
      sample_embedding_length].
    name: String naming the layer.
    model_config: A ModelConfig object.

  Returns:
    Tensor with shape [shape_embedding_length].
  """
  with tf.variable_scope(name):
    sample_embedding_length = sample_embeddings.get_shape().as_list()[1]
    beta = unbatched_fc_layer(shape_embedding, sample_embedding_length,
                              'beta_fc')
    gamma = unbatched_fc_layer(shape_embedding, sample_embedding_length,
                               'gamma_fc')
    batch_mean, batch_variance = tf.nn.moments(sample_embeddings, axes=[0, 1])
    ensure_is_scalar(batch_mean)
    ensure_is_scalar(batch_variance)
    running_mean = tf.get_variable(
        'running_mean',
        shape=[1],
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=False)
    running_variance = tf.get_variable(
        'running_variance',
        shape=[1],
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=False)
    is_training = model_config.train
    if is_training:
      running_mean = tf.assign(running_mean,
                               0.995 * running_mean + 0.005 * batch_mean)
      running_variance = tf.assign(
          running_variance, 0.995 * running_variance + 0.005 * batch_variance)
      with tf.control_dependencies([running_mean, running_variance]):
        mean = tf.identity(running_mean)
        variance = tf.identity(running_variance)
    else:
      mean = running_mean
      variance = running_variance

    denom = tf.sqrt(variance + SQRT_EPS)
    out = gamma * tf.divide((sample_embeddings - mean), denom) + beta
    return out


def batched_occnet_resnet_layer(shape_embedding, sample_embeddings, name,
                                model_config):
  """Applies a fully connected resnet layer to the input.

  Args:
    shape_embedding: Tensor with shape [batch_size, shape_embedding_length].
    sample_embeddings: Tensor with shape [batch_size, sample_count,
      sample_embedding_length].
    name: String name for the variable scope of the layer.
    model_config: A ModelConfig object.

  Returns:
    Tensor with shape [sample_count, sample_embedding_length].
  """
  with tf.variable_scope(name):
    sample_embedding_length = sample_embeddings.get_shape().as_list()[2]
    init_sample_embeddings = sample_embeddings
    sample_embeddings = batched_cbn_layer(shape_embedding, sample_embeddings,
                                          'cbn_1', model_config)
    if model_config.hparams.fon == 't':
      init_sample_embeddings = sample_embeddings
    with tf.variable_scope('fc_1'):
      sample_embeddings = tf.nn.relu(sample_embeddings)
      sample_embeddings = layers.fully_connected(
          inputs=sample_embeddings,
          num_outputs=sample_embedding_length,
          activation_fn=None,
          normalizer_fn=None,
          normalizer_params=None)
    sample_embeddings = batched_cbn_layer(shape_embedding, sample_embeddings,
                                          'cbn_2', model_config)
    with tf.variable_scope('fc_2'):
      sample_embeddings = tf.nn.relu(sample_embeddings)
      sample_embeddings = layers.fully_connected(
          inputs=sample_embeddings,
          num_outputs=sample_embedding_length,
          activation_fn=None,
          normalizer_fn=None,
          normalizer_params=None)
    return init_sample_embeddings + sample_embeddings


def occnet_resnet_layer(shape_embedding, sample_embeddings, name, model_config):
  """Applies a fully connected resnet layer to the input.

  Args:
    shape_embedding: Tensor with shape [shape_embedding_length].
    sample_embeddings: Tensor with shape [sample_count,
      sample_embedding_length].
    name: String name for the variable scope of the layer.
    model_config: A ModelConfig object.

  Returns:
    Tensor with shape [sample_count, sample_embedding_length].
  """
  with tf.variable_scope(name):
    sample_embedding_length = sample_embeddings.get_shape().as_list()[1]
    init_sample_embeddings = sample_embeddings
    sample_embeddings = cbn_layer(shape_embedding, sample_embeddings,
                                  'cbn_1', model_config)
    if model_config.hparams.fon == 't':
      init_sample_embeddings = sample_embeddings
    with tf.variable_scope('fc_1'):
      sample_embeddings = tf.nn.relu(sample_embeddings)
      sample_embeddings = layers.fully_connected(
          inputs=sample_embeddings,
          num_outputs=sample_embedding_length,
          activation_fn=None,
          normalizer_fn=None,
          normalizer_params=None)
    sample_embeddings = cbn_layer(shape_embedding, sample_embeddings,
                                  'cbn_2', model_config)
    with tf.variable_scope('fc_2'):
      sample_embeddings = tf.nn.relu(sample_embeddings)
      sample_embeddings = layers.fully_connected(
          inputs=sample_embeddings,
          num_outputs=sample_embedding_length,
          activation_fn=None,
          normalizer_fn=None,
          normalizer_params=None)
    return init_sample_embeddings + sample_embeddings


def ensure_rank(tensor, rank):
  real_shape = tensor.get_shape().as_list()
  real_rank = len(real_shape)
  if real_rank != rank:
    raise ValueError('Expected a tensor with rank %i, but given a tensor '
                     'with shape %s' % (real_rank, str(real_shape)))


def ensure_dims_match(t1, t2, dims):
  if isinstance(dims, int):
    dims = [dims]
  t1_shape = t1.get_shape().as_list()
  t2_shape = t2.get_shape().as_list()
  for dim in dims:
    if t1_shape[dim] != t2_shape[dim]:
      raise ValueError('Expected tensors %s and %s to match along dims %s' %
                       (str(t1_shape), str(t2_shape), str(dims)))


def ensure_shape(tensor, shape):
  """Raises a ValueError if the input tensor doesn't have the expected shape."""
  real_shape = tensor.get_shape().as_list()
  failing = False
  if len(real_shape) != len(shape):
    failing = True
  if not failing:
    for dim, si in enumerate(shape):
      if si != -1 and si != real_shape[dim]:
        failing = True
  if failing:
    raise ValueError('Expected tensor with shape %s to have shape %s.' %
                     (str(real_shape), str(shape)))


def one_shape_occnet_decoder(embedding, samples, apply_sigmoid,
                             model_config):
  """Computes the OccNet output for the input embedding and its sample batch.

  Args:
    embedding: Tensor with shape [shape_embedding_length].
    samples: Tensor with shape [sample_count, 3].
    apply_sigmoid: Boolean. Whether to apply a sigmoid layer to the final linear
      activations.
    model_config: A ModelConfig object.

  Returns:
    Tensor with shape [sample_count, 1].
  """
  ensure_rank(embedding, 1)
  ensure_rank(samples, 2)
  with tf.variable_scope('OccNet'):
    sample_embedding_length = model_config.hparams.ips
    resnet_layer_count = model_config.hparams.orc
    with tf.variable_scope('sample_resize_fc'):
      sample_embeddings = layers.fully_connected(
          inputs=samples,
          num_outputs=sample_embedding_length,
          activation_fn=None,
          normalizer_fn=None,
          normalizer_params=None)
    for i in range(resnet_layer_count):
      sample_embeddings = occnet_resnet_layer(embedding, sample_embeddings,
                                              'fc_resnet_layer_%i' % i,
                                              model_config)
    sample_embeddings = cbn_layer(embedding, sample_embeddings,
                                  'final_cbn', model_config)
    with tf.variable_scope('final_activation'):
      vals = layers.fully_connected(
          inputs=sample_embeddings,
          num_outputs=1,
          activation_fn=None,
          normalizer_fn=None,
          normalizer_params=None)
      if apply_sigmoid:
        vals = tf.sigmoid(vals)
    return vals


def multishape_occnet_decoder(embedding, samples, apply_sigmoid, model_config):
  """Computes the OccNet output for the input embeddings and its sample batch.

  Args:
    embedding: Tensor with shape [batch_size, shape_embedding_length].
    samples: Tensor with shape [batch_size, sample_count, 3].
    apply_sigmoid: Boolean. Whether to apply a sigmoid layer to the final linear
      activations.
    model_config: A ModelConfig object.

  Returns:
    Tensor with shape [sample_count, 1].
  """
  ensure_rank(embedding, 2)
  ensure_rank(samples, 3)
  batch_size, sample_count = samples.get_shape().as_list()[0:2]
  with tf.variable_scope('OccNet'):
    sample_embedding_length = model_config.hparams.ips
    resnet_layer_count = model_config.hparams.orc
    with tf.variable_scope('sample_resize_fc'):
      sample_embeddings = layers.fully_connected(
          inputs=samples,
          num_outputs=sample_embedding_length,
          activation_fn=None,
          normalizer_fn=None,
          normalizer_params=None)
    ensure_shape(sample_embeddings,
                 [batch_size, sample_count, sample_embedding_length])
    for i in range(resnet_layer_count):
      sample_embeddings = batched_occnet_resnet_layer(embedding,
                                                      sample_embeddings,
                                                      'fc_resnet_layer_%i' % i,
                                                      model_config)
    sample_embeddings = batched_cbn_layer(embedding, sample_embeddings,
                                          'final_cbn', model_config)
    with tf.variable_scope('final_activation'):
      vals = layers.fully_connected(
          inputs=sample_embeddings,
          num_outputs=1,
          activation_fn=None,
          normalizer_fn=None,
          normalizer_params=None)
      if apply_sigmoid:
        vals = tf.sigmoid(vals)
    return vals


def remove_batch_dim(tensor):
  shape = tensor.get_shape().as_list()
  rank = len(shape)
  ensure_shape(tensor, [1] + (rank-1)*[-1])
  tensor = tf.reshape(tensor, shape[1:])
  return tensor


def add_batch_dim(tensor):
  shape = tensor.get_shape().as_list()
  tensor = tf.reshape(tensor, [1] + shape)
  return tensor


def occnet_decoder(embedding, samples, apply_sigmoid, model_config):
  """Computes the OccNet output for the input embedding and its sample batch.

  Args:
    embedding: Tensor with shape [batch_size, shape_embedding_length].
    samples: Tensor with shape [batch_size, sample_count, 3].
    apply_sigmoid: Boolean. Whether to apply a sigmoid layer to the final linear
      activations.
    model_config: A ModelConfig object.

  Returns:
    Tensor with shape [batch_size, sample_count, 1].
  """
  if model_config.hparams.hyo == 't':
    samples = math_util.nerfify(samples, 10, flatten=True, interleave=True)
    sample_len = 60
  else:
    sample_len = 3
  if model_config.hparams.dd == 't':
    tf.logging.info(
        'BID0: Running SS Occnet Decoder with input shapes embedding=%s, samples=%s',
        repr(embedding.get_shape().as_list()),
        repr(samples.get_shape().as_list()))
    ensure_shape(embedding, [1, -1])
    ensure_shape(samples, [1, -1, 3])
    vals = one_shape_occnet_decoder(
        remove_batch_dim(embedding), remove_batch_dim(samples), apply_sigmoid,
        model_config)
    return add_batch_dim(vals)
  batch_size, embedding_length = embedding.get_shape().as_list()
  ensure_shape(embedding, [batch_size, embedding_length])
  ensure_shape(samples, [batch_size, -1, sample_len])
  # Debugging:
  vals = multishape_occnet_decoder(embedding, samples, apply_sigmoid,
                                   model_config)
  return vals


def occnet_encoder(inputs, model_config):
  ensure_rank(inputs, 5)
  return net_util.inputs_to_feature_vector(inputs, 256, model_config)

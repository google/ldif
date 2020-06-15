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
"""Example models."""

import importlib

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.nets import cnn
from ldif.nets import occnet
from ldif.nets import pointnet
from ldif.representation import structured_implicit_function
from ldif.util import geom_util
from ldif.util import net_util
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order

import tensorflow as tf

importlib.reload(occnet)
importlib.reload(cnn)
importlib.reload(pointnet)


class Observation(object):
  """An observation that is seen by a network."""

  def __init__(self, model_config, training_example):
    # Auxiliaries:
    samp = model_config.hparams.samp
    if 'p' in samp:
      # Then we have access to a point cloud as well:
      self._surface_points = training_example.all_surface_points
      samp = samp.replace('p', '')
    if 'q' in samp:
      self._surface_points = training_example.all_surface_points_from_depth
      samp = samp.replace('q', '')
    if 'n' in samp:
      # Then we have access to normals:
      if 'q' in samp:
        raise ValueError("Can't combine normals with xyz from depth.")
      self._normals = training_example.all_normals
      samp = samp.replace('n', '')
    else:
      self._normals = None
    # Main input:
    idx1 = model_config.hparams.didx
    if samp == 'imd':
      tensor = training_example.depth_renders
      im_count = 20
      channel_count = 1
    elif samp == 'imxyz':
      tensor = training_example.xyz_renders
      im_count = 20
      channel_count = 3
    elif samp == 'rgb':
      tensor = training_example.chosen_renders
      im_count = 1
      channel_count = 3
    elif samp == 'im1d':
      if ('ShapeNetNSSDodecaSparseLRGMediumSlimPC' in
          training_example.proto_name):
        tensor = training_example.depth_renders[:, idx1:idx1 + 1, ...]
      elif training_example.proto_name == 'ShapeNetOneImXyzPC' or 'Waymo' in training_example.proto_name:
        tensor = training_example.depth_render
      im_count = 1
      channel_count = 1
    elif samp == 'im1l':
      tensor = training_example.lum_renders[:, idx1:idx1 + 1, ...]
      im_count = 1
      channel_count = 1
    elif samp == 'im1xyz':
      if ('ShapeNetNSSDodecaSparseLRGMediumSlimPC' in
          training_example.proto_name):
        tensor = training_example.xyz_renders[:, idx1:idx1 + 1, ...]
      elif training_example.proto_name == 'ShapeNetOneImXyzPC' or 'Waymo' in training_example.proto_name:
        tensor = training_example.xyz_render
      im_count = 1
      channel_count = 3
    elif samp == 'imrd':
      tensor = training_example.random_depth_images
      im_count = model_config.hparams.rc
      channel_count = 1
    elif samp == 'imrxyz':
      tensor = training_example.random_xyz_render
      im_count = 1
      channel_count = 3
    elif samp == 'imrlum':
      tensor = training_example.random_lum_render
      im_count = 1
      channel_count = 1
    elif samp == 'imlum':
      im_count = 20
      channel_count = 1
      tensor = training_example.lum_renders
    else:
      raise ValueError('Unrecognized samp: %s -> %s' %
                       (model_config.hparams.samp, samp))
    tensor = tf.ensure_shape(tensor, [
        model_config.hparams.bs, im_count, model_config.hparams.h,
        model_config.hparams.w, channel_count
    ])
    self._tensor = tensor
    self._one_image_one_channel_tensor = None
    self._model_config = model_config
    self._samp = samp

  @property
  def tensor(self):
    return self._tensor

  @property
  def surface_points(self):
    return self._surface_points

  @property
  def normals(self):
    return self._normals

  @property
  def one_image_one_channel_tensor(self):
    """A summary that describes the observation with a single-channel."""
    # For now just return a fixed image and the first channel.
    # TODO(kgenova) Check if RGB, convert to luminance, etc., maybe tile and
    # resize to fit a grid as a single image.
    if self._one_image_one_channel_tensor is None:
      im_count = self.tensor.get_shape().as_list()[1]
      if im_count == 1:
        self._one_image_one_channel_tensor = self.tensor[:, 0, :, :, 0:1]
      else:
        self._one_image_one_channel_tensor = self.tensor[:, 1, :, :, 0:1]
    return self._one_image_one_channel_tensor


class Prediction(object):
  """A prediction made by a model."""

  def __init__(self, model_config, observation, structured_implicit, embedding):
    self._observation = observation
    self._structured_implicit = structured_implicit
    self._embedding = embedding
    self._in_out_image = None
    self._model_config = model_config

  def has_embedding(self):
    return self._embedding is not None

  def export_signature_def(self):
    input_map = {}
    input_map['observation'] = self.observation.tensor
    output_map = {}
    if self.has_embedding:  # pylint:disable=using-constant-test
      output_map['embedding'] = self.embedding
    output_map['structured_implicit_vector'] = self.structured_implicit.vector
    return tf.saved_model.signature_def_utils.predict_signature_def(
        inputs=input_map, outputs=output_map)

  @property
  def embedding(self):
    return self._embedding

  @property
  def structured_implicit(self):
    return self._structured_implicit

  @property
  def observation(self):
    return self._observation


def pointnet_sif_predictor(observation, element_count, element_length,
                           model_config):
  """A cnn that maps 1+ images with 1+ chanels to a feature vector."""
  inputs = tf.concat([observation.surface_points, observation.normals], axis=-1)
  batch_size = inputs.get_shape().as_list()[0]
  sample_count = inputs.get_shape().as_list()[1]
  max_encodable = 1024
  if sample_count > max_encodable:
    sample_indices = tf.random.uniform(
        [batch_size, sample_count],
        minval=0,
        maxval=sample_count - 1,
        dtype=tf.int32)
    inputs = tf.batch_gather(inputs, sample_indices)
  embedding = point_encoder(inputs, element_count * element_length,
                            model_config)
  batch_size = inputs.get_shape().as_list()[0]
  prediction = tf.reshape(embedding,
                          [batch_size, element_count, element_length])
  return prediction, embedding


# From https://stackoverflow.com/questions/34945554/
# how-to-set-layer-wise-learning-rate-in-tensorflow
def lr_mult(alpha):
  """Decreases the learning rate update by a factor of alpha."""

  @tf.custom_gradient
  def _lr_mult(x):

    def grad(dy):
      return dy * alpha * tf.ones_like(x)

    return x, grad

  return _lr_mult


def point_encoder(points, output_dimensionality, model_config):
  """Encodes a point cloud (either with or without normals) to an embedding."""
  assert len(points.shape) == 3
  # TODO(kgenova) This could reshape to the batch dimension to support
  batch_size = points.get_shape().as_list()[0]
  # [..., N, 3] inputs.
  if model_config.hparams.pe == 'pn':
    if model_config.hparams.udp == 't':
      pointnet_fun = pointnet.pointnet_depr
      kwargs = {}
    else:
      pointnet_fun = pointnet.pointnet
      kwargs = {'use_gpu': model_config.train}
    embedding = pointnet_fun(
        points,
        output_feature_count=output_dimensionality,
        apply_learned_64d_tx=model_config.hparams.p64 == 't',
        use_bad_reduce=model_config.hparams.fbp == 'f',
        nerfify=model_config.hparams.hyp == 't',
        maxpool_feature_count=model_config.hparams.mfc,
        **kwargs)
  elif model_config.hparams.pe == 'dg':
    raise ValueError('DGCNN is no longer supported.')
  embedding = tf.reshape(embedding, [batch_size, output_dimensionality])
  if len(embedding.shape) != 2 or embedding.get_shape().as_list(
  )[-1] != output_dimensionality:
    raise ValueError(
        f'Unexpected output shape: {embedding.get_shape().as_list()}')
  return embedding


class StructuredImplicitModel(object):
  """An instance of a network that predicts a structured implicit function."""

  def __init__(self, model_config, name):
    if model_config.hparams.arch == 'efcnn':
      self.inference_fun = cnn.early_fusion_cnn
    elif model_config.hparams.arch == 'mfcnn':
      self.inference_fun = cnn.mid_fusion_cnn
    elif model_config.hparams.arch == 'pn':
      self.inference_fun = pointnet_sif_predictor
    else:
      raise ValueError(
          'Invalid StructuredImplicitModel architecture hparam: %s' %
          model_config.hparams.arch)
    if model_config.hparams.ipe in ['t', 'e']:
      self.single_element_implicit_eval_fun = occnet.occnet_decoder
    else:
      self.single_element_implicit_eval_fun = None
    self._model_config = model_config
    self._name = name
    self._forward_call_count = 0
    self._eval_implicit_parameters_call_count = 0
    self._enable_deprecated = False

  @property
  def name(self):
    return self._name

  def _global_local_forward(self, observation):
    """A forward pass that include both template and element inference."""
    with tf.name_scope(self._name + '/forward'):
      reuse = self._forward_call_count > 0
      explicit_element_length = structured_implicit_function.element_explicit_dof(
          self._model_config)
      implicit_embedding_length = structured_implicit_function.element_implicit_dof(
          self._model_config)
      element_count = self._model_config.hparams.sc
      if explicit_element_length <= 0:
        raise ValueError('Invalid element length. Embedding has length '
                         '%i, but total length is only %i.' %
                         (implicit_embedding_length, explicit_element_length))
      with tf.variable_scope(self._name + '/forward', reuse=reuse):
        with tf.variable_scope('explicit_embedding_cnn'):
          explicit_parameters, explicit_embedding = self.inference_fun(
              observation, element_count, explicit_element_length,
              self._model_config)
          if self._model_config.hparams.elr != 1.0:
            explicit_parameters = lr_mult(self._model_config.hparams.elr)(
                explicit_parameters)
      sif = structured_implicit_function.StructuredImplicit.from_activation(
          self._model_config, explicit_parameters, self)
      # Now we can compute world2local
      world2local = sif.world2local

      if implicit_embedding_length > 0:
        with tf.variable_scope(self._name + '/forward', reuse=reuse):
          with tf.variable_scope('implicit_embedding_net'):
            local_points, local_normals, _, _ = geom_util.local_views_of_shape(
                observation.surface_points,
                world2local,
                local_point_count=self._model_config.hparams.lpc,
                global_normals=observation.normals)
            # Output shapes are both [B, EC, LPC, 3].
            if 'n' not in self._model_config.hparams.samp:
              flat_point_features = tf.reshape(local_points, [
                  self._model_config.hparams.bs * self._model_config.hparams.sc,
                  self._model_config.hparams.lpc, 3
              ])
            else:
              flat_point_features = tf.reshape(
                  tf.concat([local_points, local_normals], axis=-1), [
                      self._model_config.hparams.bs *
                      self._model_config.hparams.sc,
                      self._model_config.hparams.lpc, 6
                  ])
            encoded_iparams = point_encoder(flat_point_features,
                                            self._model_config.hparams.ips,
                                            self._model_config)
            iparams = tf.reshape(encoded_iparams, [
                self._model_config.hparams.bs, self._model_config.hparams.sc,
                self._model_config.hparams.ips
            ])
        sif.set_iparams(iparams)
        embedding = tf.concat([
            explicit_embedding,
            tf.reshape(iparams, [self._model_config.hparams.bs, -1])
        ],
                              axis=-1)
      else:
        embedding = explicit_embedding
      self._forward_call_count += 1
      return Prediction(self._model_config, observation, sif, embedding)

  def forward(self, observation):
    """Evaluates the explicit and implicit parameter vectors as a Prediction."""
    if self._model_config.hparams.ia == 'p':
      return self._global_local_forward(observation)
    with tf.name_scope(self._name + '/forward'):
      reuse = self._forward_call_count > 0
      tf.logging.info('Call #%i to %s.forward().', self._forward_call_count,
                      self._name)
      element_count = self._model_config.hparams.sc
      flat_element_length = structured_implicit_function.element_dof(
          self._model_config)
      with tf.variable_scope(self._name + '/forward', reuse=reuse):
        if self._model_config.hparams.ia == '1':
          structured_implicit_activations, embedding = self.inference_fun(
              observation, element_count, flat_element_length,
              self._model_config)
        elif self._model_config.hparams.ia == '2':
          structured_implicit_activations, embedding = net_util.twin_inference(
              self.inference_fun, observation, element_count,
              flat_element_length, self._model_config)
        else:
          raise ValueError('Invalid value for hparam ia: %i' %
                           self._model_config.hparams.ia)
        if self._model_config.hparams.elr != 1.0:
          structured_implicit_activations = lr_mult(
              self._model_config.hparams.elr)(
                  structured_implicit_activations)
      self._forward_call_count += 1
      structured_implicit = (
          structured_implicit_function.StructuredImplicit.from_activation(
              self._model_config, structured_implicit_activations, self))
    return Prediction(self._model_config, observation, structured_implicit,
                      embedding)

  def eval_implicit_parameters(self, implicit_parameters, samples):
    """Decodes each implicit parameter vector at each of its sample points.

    Args:
      implicit_parameters: Tensor with shape [batch_size, element_count,
        element_embedding_length]. The embedding associated with each element.
      samples: Tensor with shape [batch_size, element_count, sample_count, 3].
        The sample locations. Each embedding vector will be decoded at each of
        its sample locations.

    Returns:
      Tensor with shape [batch_size, element_count, sample_count, 1]. The
        decoded value of each element's embedding at each of the samples for
        that embedding.
    """
    # Each element has its own network:
    if self.single_element_implicit_eval_fun is None:
      raise ValueError('The implicit decoder function is None.')
    implicit_param_shape_in = implicit_parameters.get_shape().as_list()
    tf.logging.info('BID0: Input implicit param shape: %s',
                    repr(implicit_param_shape_in))
    tf.logging.info('BID0: Input samples shape: %s',
                    repr(samples.get_shape().as_list()))
    reuse = self._eval_implicit_parameters_call_count > 0
    # TODO(kgenova) Now that batching OccNet is supported, batch this call.
    with tf.variable_scope(
        self._name + '/eval_implicit_parameters', reuse=reuse):
      if self._enable_deprecated:
        log.info('Deprecated eval.')
        vals = self._deprecated_multielement_eval(implicit_parameters, samples)
      else:
        batch_size, element_count, element_embedding_length = (
            implicit_parameters.get_shape().as_list())
        sample_count = samples.get_shape().as_list()[-2]
        batched_parameters = tf.reshape(
            implicit_parameters,
            [batch_size * element_count, element_embedding_length])
        batched_samples = tf.reshape(
            samples, [batch_size * element_count, sample_count, 3])
        if self._model_config.hparams.npe == 't':
          raise ValueError(
              'Incompatible hparams. Must use _deprecated_multielement_eval'
              'if requesting separate network weights per shape element.')
        with tf.variable_scope('all_elements', reuse=False):
          batched_vals = self.single_element_implicit_eval_fun(
              batched_parameters,
              batched_samples,
              apply_sigmoid=False,
              model_config=self._model_config)
        vals = tf.reshape(batched_vals,
                          [batch_size, element_count, sample_count, 1])
    self._eval_implicit_parameters_call_count += 1
    return vals

  def _deprecated_multielement_eval(self, implicit_parameters, samples):
    """An eval provided for backwards compatibility."""
    vallist = []
    implicit_parameter_list = tf.unstack(implicit_parameters, axis=1)
    sample_list = tf.unstack(samples, axis=1)
    tf.logging.info('BID0: Call #%i to %s.eval_implicit_parameters().',
                    self._eval_implicit_parameters_call_count, self._name)
    tf.logging.info('BID0: List length: %i', len(implicit_parameter_list))
    for i in range(len(implicit_parameter_list)):
      scope = 'all_elements'
      if self._model_config.hparams.npe == 't':
        scope = 'element_%i' % i
        reuse_elements = False
      else:
        reuse_elements = i > 0
      tf.logging.info('BID0: Building eval_implicit_parameters for element %i',
                      i)
      with tf.variable_scope(scope, reuse=reuse_elements):
        vallist.append(
            self.single_element_implicit_eval_fun(
                implicit_parameter_list[i],
                sample_list[i],
                apply_sigmoid=False,
                model_config=self._model_config))
    return tf.stack(vallist, axis=1)

  def as_placeholder(self):
    """Creates a doppleganger StructuredImplicitModel with tf.placeholders."""
    # Right now there are no input tensors, so a new model is fine.
    placeholder = StructuredImplicitModel(self._model_config, self._name)
    # We set the call count for the doppleganger to the current call count,
    # which is necessary so that calls to the placeholder class reuse or don't
    # reuse variables appropriately.
    # This isn't really an outside access since it's in a method for that class:
    # pylint: disable=protected-access
    placeholder._eval_implicit_parameters_call_count = (
        self._eval_implicit_parameters_call_count)
    placeholder._forward_call_count = self._forward_call_count
    # pylint: enable=protected-access
    return placeholder

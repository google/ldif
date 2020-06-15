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
"""A class for an instance of a structured implicit function."""

import math
import time

import numpy as np
import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import geom_util
from ldif.util import sdf_util
from ldif.util import tf_util
from ldif.util import camera_util
from ldif.util import np_util

from ldif.representation import quadrics

from ldif.util.file_util import log
# pylint: enable=g-bad-import-order

RADII_EPS = 1e-10


def _unflatten(model_config, vector):
  """Given a flat shape vector, separates out the individual parameter sets."""
  radius_shape = element_radius_shape(model_config)
  radius_len = int(np.prod(radius_shape))
  assert len(radius_shape) == 1  # If the radius is a >1 rank tensor, unpack it.
  explicit_param_count = 1  # A single constant.
  # First, determine if there are implicit parameters present.
  total_explicit_length = explicit_param_count + 3 + radius_len
  provided_length = vector.get_shape().as_list()[-1]
  # log.info('Provided len, explicit len: %i, %i' %
  # (provided_length, total_explicit_length))
  if provided_length > total_explicit_length:
    expected_implicit_length = model_config.hparams.ips
    leftover_length = provided_length - total_explicit_length
    if leftover_length != expected_implicit_length:
      raise ValueError(('Unable to interpret input vector as either explicit or'
                        ' implicit+explicit vector. Shape: %s') %
                       repr(vector.get_shape().as_list()))
    constant, center, radius, iparams = tf.split(
        vector, [explicit_param_count, 3, radius_len, expected_implicit_length],
        axis=-1)
  elif provided_length == total_explicit_length:
    constant, center, radius = tf.split(
        vector, [explicit_param_count, 3, radius_len], axis=-1)
    iparams = None
  else:
    raise ValueError('Too few input parameters for even explicit vector: %s' %
                     repr(vector.get_shape().as_list()))
  return constant, center, radius, iparams


def element_explicit_dof(model_config):
  """Determines the number of (analytic) degrees of freedom in a single blob."""
  # Constant:
  constant_dof = 1

  # Center:
  center_dof = 3  # in R^3

  # Radius:
  if model_config.hparams.r == 'iso':
    radius_dof = 1
  elif model_config.hparams.r == 'aa':
    radius_dof = 3
  elif model_config.hparams.r == 'cov':
    radius_dof = 6
  else:
    raise ValueError('Unrecognized radius hyperparameter: %s' %
                     model_config.hparams.r)
  return constant_dof + center_dof + radius_dof

  # Implicit Parameters:


  # if model_config.hparams.ip == 'no':
  #   ip_dof = 0
  # elif model_config.hparams.ip == 'sb':
  #   ip_dof = model_config.hparams.ips
  # else:
  #   raise ValueError('Unrecognized implicit parameter hyperparameter: %s' %
  #                    model_config.hparams.ip)
def element_implicit_dof(model_config):
  if model_config.hparams.ipe != 'f':
    return model_config.hparams.ips
  return 0


def element_dof(model_config):
  """Returns the DoF of a single shape element."""
  return element_explicit_dof(model_config) + element_implicit_dof(model_config)


def element_constant_shape(model_config):
  # We will need the model config in the future if we enable other options:
  # pylint: disable=unused-argument
  return [1]


def element_center_shape(model_config):
  # We will need the model config in the future if we enable other options:
  # pylint: disable=unused-argument
  return [3]


def element_radius_shape(model_config):
  if model_config.hparams.r == 'iso':
    return [1]
  elif model_config.hparams.r == 'aa':
    return [3]
  elif model_config.hparams.r == 'cov':
    return [6]
  else:
    raise ValueError('Unrecognized radius hyperparameter: %s' %
                     model_config.hparams.r)


def element_iparam_shape(model_config):
  if model_config.hparams.ip == 'no':
    return []
  elif model_config.hparams.ip == 'bp':
    return [model_config.hparams.ips]
  else:
    raise ValueError('Unrecognized implicit parameter hyperparameter: %s' %
                     model_config.hparams.ip)


def ensure_net_if_needed(model_config, net):
  if model_config.hparams.ip != 'no' and net is None:
    raise ValueError('Must provide network for sample decoding when using '
                     'iparams (hparam ip=%s).' % model_config.hparams.ip)


def sigma(x, a):
  return x / tf.sqrt(a * a + x * x)


def ensure_np_shape(arr, expected_shape, name):
  """Verifies a numpy array matches the expected shape. Supports symbols."""
  shape_ok = len(arr.shape) == len(expected_shape)
  if shape_ok:
    symbols = {}
    for i in range(len(arr.shape)):
      if isinstance(expected_shape[i], int) and expected_shape[i] != -1:
        shape_ok = shape_ok and arr.shape[i] == expected_shape[i]
      elif expected_shape[i] == -1:
        continue
      else:
        if expected_shape[i] in symbols:
          shape_ok = shape_ok and arr.shape[i] == symbols[expected_shape[i]]
        else:
          symbols[expected_shape[i]] = arr.shape[i]
  if not shape_ok:
    raise ValueError(
        'Expected numpy array %s to have shape %s but it has shape %s.' %
        (name, str(expected_shape), str(arr.shape)))


class StructuredImplicitNp(object):
  """A (batch of) numpy structured implicit functions(s)."""

  def __init__(self, structured_implicit):
    """Builds out the tensorflow graph portions that are needed."""
    self._initialized = False
    self._structured_implicit_tf = structured_implicit
    self._structured_implicit_ph = structured_implicit.as_placeholder()
    self._build_sample_on_grid()

  def initialize(self, session, feed_dict):
    """Initializes the data by evaluating the tensors in the session."""
    np_list = session.run(
        self._structured_implicit_tf.tensor_list, feed_dict=feed_dict)
    (self.element_constants, self.element_centers,
     self.element_radii) = np_list[:3]
    # pylint:disable=protected-access
    if self._structured_implicit_tf._model_config.hparams.ipe != 'f':
      # pylint:enable=protected-access
      # if len(np_list) == 4:
      self.element_iparams = np_list[3]
    else:
      if len(np_list) == 4:
        log.info('Warning: implicit parameters present but not enabled.'
                 ' Eliding them to avoid using untrained values.')
      self.element_iparams = None
    self._session = session
    self._initialized = True

  def ensure_initialized(self):
    if not self._initialized:
      raise AssertionError(
          'A StructuredImplicitNp must be initialized before use.')

  def flat_vector(self):
    self.ensure_initialized()
    l = [self.element_constants, self.element_centers, self.element_radii]
    # if self._structured_implicit_tf._model_config.hparams.ipe == 't':
    if self.element_iparams is not None:
      l.append(self.element_iparams)
    return np.concatenate(l, axis=-1)

  def _feed_dict(self):
    base_feed_dict = {
        self._structured_implicit_ph.element_constants: self.element_constants,
        self._structured_implicit_ph.element_centers: self.element_centers,
        self._structured_implicit_ph.element_radii: self.element_radii
    }
    if self.element_iparams is not None:
      base_feed_dict[
          self._structured_implicit_ph.element_iparams] = self.element_iparams
    return base_feed_dict

  def sample_on_grid(self, sample_center, sample_size, resolution):
    """Evaluates the function on a grid."""
    self.ensure_initialized()
    # ensure_np_shape(sample_grid, ['res', 'res', 'res'], 'sample_grid')
    # mcubes_res = sample_grid.shape[0]
    if resolution % self._block_res:
      raise ValueError(
          'Input resolution is %i, but must be a multiple of block size, %i.' %
          (resolution, self._block_res))
    block_count = resolution // self._block_res
    block_size = sample_size / block_count

    base_grid = np_util.make_coordinate_grid_3d(
        length=self._block_res,
        height=self._block_res,
        width=self._block_res,
        is_screen_space=False,
        is_homogeneous=False).astype(np.float32)
    lower_corner = sample_center - sample_size / 2.0
    start_time = time.time()
    l_block = []
    i = 0
    for li in range(block_count):
      l_min = lower_corner[2] + li * block_size
      h_block = []
      for hi in range(block_count):
        h_min = lower_corner[1] + hi * block_size
        w_block = []
        for wi in range(block_count):
          w_min = lower_corner[0] + wi * block_size
          offset = np.reshape(
              np.array([w_min, l_min, h_min], dtype=np.float32), [1, 1, 1, 3])
          sample_locations = block_size * base_grid + offset
          feed_dict = self._feed_dict()
          feed_dict[self._sample_locations_ph] = sample_locations
          grid_out_np = self._session.run(
              self._predicted_class_grid, feed_dict=feed_dict)
          i += 1
          w_block.append(grid_out_np)
        h_block.append(np.concatenate(w_block, axis=2))
      l_block.append(np.concatenate(h_block, axis=0))
    grid_out = np.concatenate(l_block, axis=1)
    compute_time = time.time() - start_time
    log.info('Post Initialization Time: %s' % compute_time)
    return grid_out, compute_time

  def _build_sample_on_grid(self):
    """Builds the graph nodes associated with sample_on_grid()."""
    block_res = 32
    self._block_res = block_res

    self._sample_locations_ph = tf.placeholder(
        tf.float32, shape=[block_res, block_res, block_res, 3])
    samples = tf.reshape(self._sample_locations_ph, [1, block_res**3, 3])
    predicted_class, _ = self._structured_implicit_ph.class_at_samples(samples)
    self._predicted_class_grid = tf.reshape(predicted_class,
                                            [block_res, block_res, block_res])

  def extract_mesh(self):
    """Computes a mesh from the representation."""
    self.ensure_initialized()

  def render(self):
    """Computes an image of the representation."""
    self.ensure_initialized()

  def element_constants_as_quadrics(self):
    constants = self.element_constants
    quadric_shape = list(constants.shape[:-1]) + [4, 4]
    qs = np.zeros(quadric_shape, dtype=np.float32)
    qs[..., 3, 3] = np.squeeze(constants)
    return qs

  def write_to_disk(self, fnames):
    """Writes the representation to disk in the GAPS format."""
    self.ensure_initialized()
    eval_set_size = self.element_centers.shape[0]
    qs = self.element_constants_as_quadrics()
    for ei in range(eval_set_size):
      flat_quadrics = np.reshape(qs[ei, ...], [-1, 4 * 4])
      flat_centers = np.reshape(self.element_centers[ei, ...], [-1, 3])
      if self.element_radii.shape[-1] == 3:
        flat_radii = np.reshape(np.sqrt(self.element_radii[ei, ...]), [-1, 3])
      elif self.element_radii.shape[-1] == 6:
        flat_radii = np.reshape(self.element_radii[ei, ...], [-1, 6])
        flat_radii[..., :3] = np.sqrt(flat_radii[..., :3])
      if self.element_iparams is None:
        flat_params = np.concatenate([flat_quadrics, flat_centers, flat_radii],
                                     axis=1)
      else:
        flat_iparams = np.reshape(self.element_iparams[ei, ...], [-1, 32])
        flat_params = np.concatenate(
            [flat_quadrics, flat_centers, flat_radii, flat_iparams], axis=1)
      np.savetxt(fnames[ei], flat_params)


def homogenize(m):
  """Adds homogeneous coordinates to a [..., N,N] matrix."""
  batch_rank = len(m.get_shape().as_list()) - 2
  batch_dims = m.get_shape().as_list()[:-2]
  n = m.get_shape().as_list()[-1]
  assert m.get_shape().as_list()[-2] == n
  right_col = np.zeros(batch_dims + [3, 1], dtype=np.float32)
  m = tf.concat([m, right_col], axis=-1)
  lower_row = np.pad(
      np.zeros(batch_dims + [1, 3], dtype=np.float32),
      [(0, 0)] * batch_rank + [(0, 0), (0, 1)],
      mode='constant',
      constant_values=1.0)
  lower_row = tf.constant(lower_row)
  return tf.concat([m, lower_row], axis=-2)


def symgroup_equivalence_classes(model_config):
  """Generates the effective element indices for each symmetry group.

  Args:
    model_config: A ModelConfig object.

  Returns:
    A list of lists. Each sublist contains indices in the range [0,
      effective_element_count-1]. These indices map into tensors with dimension
      [..., effective_element_count, ...]. Each index appears in exactly one
      sublist, and the sublists are sorted in ascending order.
  """
  # Populate with the root elements.
  lists = [[i] for i in range(model_config.hparams.sc)]

  left_right_sym_count = model_config.hparams.lyr
  if left_right_sym_count:
    start_idx = model_config.hparams.sc
    for i in range(left_right_sym_count):
      idx = start_idx + i
      lists[i].extend(idx)
  return lists


def symgroup_class_ids(model_config):
  """Generates the equivalence class identifier for each effective element.

  Args:
    model_config: A ModelConfig object.

  Returns:
    A list of integers of length effective_element_count. Each element points to
    the 'true' (i.e. < shape count) identifier for each effective element
  """
  l = list(range(model_config.hparams.sc))

  left_right_sym_count = model_config.hparams.lyr
  if left_right_sym_count:
    l.extend(list(range(left_right_sym_count)))
  return l


def _tile_for_symgroups(model_config, elements):
  """Tiles an input tensor along its element dimension based on symmetry.

  Args:
    model_config: A ModelConfig object.
    elements: Tensor with shape [batch_size, element_count, ...].

  Returns:
    Tensor with shape [batch_size, element_count + tile_count, ...]. The
    elements have been tiled according to the model configuration's symmetry
    group description.
  """
  left_right_sym_count = model_config.hparams.lyr
  assert len(elements.get_shape().as_list()) >= 3
  # The first K elements get reflected with left-right symmetry (z-axis) as
  # needed.
  if left_right_sym_count:
    first_k = elements[:, :left_right_sym_count, ...]
    elements = tf.concat([elements, first_k], axis=1)
  # TODO(kgenova) As additional symmetry groups are added, add their tiling.
  return elements


def get_effective_element_count(model_config):
  return model_config.hparams.sc + model_config.hparams.lyr


def _generate_symgroup_samples(model_config, samples):
  """Duplicates and transforms samples as needed for symgroup queries.

  Args:
    model_config: A ModelConfig object.
    samples: Tensor with shape [batch_size, sample_count, 3].

  Returns:
    Tensor with shape [batch_size, effective_element_count, sample_count, 3].
  """
  assert len(samples.get_shape().as_list()) == 3
  samples = tf_util.tile_new_axis(
      samples, axis=1, length=model_config.hparams.sc)

  left_right_sym_count = model_config.hparams.lyr

  if left_right_sym_count:
    first_k = samples[:, :left_right_sym_count, :]
    first_k = geom_util.z_reflect(first_k)
    samples = tf.concat([samples, first_k], axis=1)
  return samples


def constants_to_quadrics(constants):
  """Convert a set of constants to quadrics.

  Args:
    constants: Tensor with shape [..., 1].

  Returns:
    quadrics: Tensor with shape [..., 4,4]. All entries except the
    bottom-right corner are 0. That corner is the constant.
  """
  zero = tf.zeros_like(constants)
  last_row = tf.concat([zero, zero, zero, constants], axis=-1)
  zero_row = tf.zeros_like(last_row)
  return tf.stack([zero_row, zero_row, zero_row, last_row], axis=-2)


def _transform_samples(samples, tx):
  """Applies a 4x4 transformation to XYZ coordinates.

  Args:
    samples: Tensor with shape [..., sample_count, 3].
    tx: Tensor with shape [..., 4, 4].

  Returns:
    Tensor with shape [..., sample_count, 3]. The input samples in a new
    coordinate frame.
  """
  # We assume the center is an XYZ position for this transformation:
  samples = geom_util.to_homogeneous(samples, is_point=True)
  samples = tf.matmul(samples, tx, transpose_b=True)
  return samples[..., :3]


class StructuredImplicit(object):
  """A (batch of) structured implicit function(s)."""

  def __init__(self, model_config, constants, centers, radii, iparams, net):
    batching_dims = constants.get_shape().as_list()[:-2]
    batching_rank = len(batching_dims)
    if batching_rank == 0:
      constants = tf.expand_dims(constants, axis=0)
      radii = tf.expand_dims(radii, axis=0)
      centers = tf.expand_dims(centers, axis=0)
    self._constants = constants
    self._radii = radii
    self._centers = centers
    self._iparams = iparams
    self._model_config = model_config
    self._packed_vector = None
    self._flat_vector = None
    self._quadrics = None
    self._net = net
    self._world2local = None

  @classmethod
  def from_packed_vector(cls, model_config, packed_vector, net):
    """Parse an already packed vector (NOT a network activation)."""
    ensure_net_if_needed(model_config, net)
    constant, center, radius, iparam = _unflatten(model_config, packed_vector)
    return cls(model_config, constant, center, radius, iparam, net)

  @classmethod
  def from_activation(cls, model_config, activation, net):
    """Parse a network activation into a structured implicit function."""
    ensure_net_if_needed(model_config, net)
    constant, center, radius, iparam = _unflatten(model_config, activation)

    if model_config.hparams.cp == 'a':
      constant = -tf.abs(constant)
    elif model_config.hparams.cp == 's':
      constant = tf.sigmoid(constant) - 1
    radius_var = tf.sigmoid(radius[..., 0:3])
    max_blob_radius = 0.15
    radius_var *= max_blob_radius
    radius_var = radius_var * radius_var
    if model_config.hparams.r == 'cov':
      # radius_rot = sigma(radius[..., 3:], 50.0)
      max_euler_angle = np.pi / 4.0
      radius_rot = tf.clip_by_value(radius[..., 3:], -max_euler_angle,
                                    max_euler_angle)
      # radius_rot *= max_euler_angle
      radius = tf.concat([radius_var, radius_rot], axis=-1)
    else:
      radius = radius_var
    center = center / 2.0
    return cls(model_config, constant, center, radius, iparam, net)

  def force_valid_values(self):
    self._constants = tf.minimum(self._constants, -1e-10)
    if self._model_config.hparams.r == 'cov':
      axisr, rotr = tf.split(self._radii, [3, 3], axis=-1)
      axisr = tf.maximum(axisr, 1e-9)
      rotr = tf.clip_by_value(rotr, -np.pi / 4.0, np.pi / 4.0)
      self._radii = tf.concat([axisr, rotr], axis=-1)
    else:
      assert self._model_config.hparams.r == 'aa'
      self._radii = tf.maximum(self._radii, 1e-9)

  @property
  def vector(self):
    """A vector with shape [batch_size, element_count, element_length]."""
    if self._packed_vector is None:
      to_pack = [self._constants, self._centers, self._radii]
      if self._iparams is not None:
        to_pack.append(self._iparams)
      self._packed_vector = tf.concat(to_pack, axis=-1)
    return self._packed_vector

  @property
  def flat_vector(self):
    """A flattened vector with shape [batch_size, -1]."""
    if self._flat_vector is None:
      sc, sd = self.vector.get_shape().as_list()[-2:]
      self._flat_vector = tf.reshape(self.vector,
                                     [self._model_config.hparams.bs, sc, sd])
    return self._flat_vector

  @property
  def net(self):
    return self._net

  @property
  def element_constants(self):
    return self._constants

  @property
  def element_centers(self):
    return self._centers

  @property
  def element_radii(self):
    return self._radii

  @property
  def element_iparams(self):
    return self._iparams

  @property
  def constant_shape(self):
    return self._constants.get_shape().as_list()[2:]

  @property
  def center_shape(self):
    return self._centers.get_shape().as_list()[2:]

  @property
  def radius_shape(self):
    return self._radii.get_shape().as_list()[2:]

  @property
  def iparam_shape(self):
    if self._iparams is None:
      return None
    else:
      return self._iparams.get_shape().as_list()[2:]

  @property
  def tensor_list(self):
    return [
        x for x in [self._constants, self._centers, self._radii, self._iparams]
        if x is not None
    ]

  @property
  def batch_size(self):
    return self._constants.get_shape().as_list()[0]

  @property
  def element_count(self):
    return self._constants.get_shape().as_list()[1]

  @property
  def constants_as_quadrics(self):
    if self._quadrics is None:
      self._quadrics = constants_to_quadrics(self._constants)
    return self._quadrics

  def zero_constants_by_threshold(self, threshold):
    """Zeros out constants 'under' (>=) the threshold in the representation.

    This is useful for removing tiny artifacts in the reconstruction.

    Args:
      threshold: A float, scalar numpy array, or scalar tensor.

    Returns:
      No return value.
    """
    self._constants = tf.where(self._constants >= threshold,
                               tf.zeros_like(self._constants), self._constants)

  def zero_constants_by_volume(self, volume_threshold):
    """Zeros out constants based on the volume of the associated ellipsoid.

    This is useful for removing tiny artifacts in the reconstruction.

    Args:
      volume_threshold: A threshold on the ellipsoid 'volume.' This is the
        volume of the ellipsoid at 1 (sqrt) radius length.

    Returns:
      No return.
    """
    sqrt_rads = tf.sqrt(tf.maximum(self._radii[..., 0:3], 0.0))
    volumes = (4.0 / 3.0 * math.pi) * tf.math.reduce_prod(
        sqrt_rads, axis=-1, keepdims=True)
    should_zero = volumes < volume_threshold
    self._constants = tf.where(should_zero, tf.zeros_like(self._constants),
                               self._constants)

  def as_placeholder(self):
    """Creates a doppleganger StructuredImplicit with tf.placeholders."""
    batch_size = self.batch_size
    element_count = self.element_count
    constants_ph = tf.placeholder(
        tf.float32, shape=[batch_size, element_count] + self.constant_shape)
    centers_ph = tf.placeholder(
        tf.float32, shape=[batch_size, element_count] + self.center_shape)
    radii_ph = tf.placeholder(
        tf.float32, shape=[batch_size, element_count] + self.radius_shape)
    if self._iparams is None:
      iparams_ph = None
    else:
      iparams_ph = tf.placeholder(
          tf.float32, shape=[batch_size, element_count] + self.iparam_shape)
    return StructuredImplicit(self._model_config, constants_ph, centers_ph,
                              radii_ph, iparams_ph, self._net)

  def set_iparams(self, iparams):
    """Adds LDIF embeddings to the SIF object."""
    input_shape = iparams.get_shape().as_list()
    expected_batch_dims = self.element_radii.get_shape().as_list()[:-1]
    expected_shape = expected_batch_dims + [self._model_config.hparams.ips]
    if len(input_shape) != len(expected_shape):
      raise ValueError(
          'Trying to set iparams with incorrect rank: %s in but %s expected.' %
          (repr(input_shape), repr(expected_shape)))
    for di, de in zip(input_shape, expected_shape):
      if di != de:
        raise ValueError(
            'Trying to set iparams with incorrect shape: %s in but %s expected.'
            % (repr(input_shape), repr(expected_shape)))
    self._iparams = iparams

  @property
  def world2local(self):
    """The world2local transformations for each element. Shape [B, EC, 4, 4]."""
    if self._world2local is None:
      self._world2local = self._compute_world2local()
    return self._world2local

  def _compute_world2local(self):
    """Computes a transformation to the local element frames for encoding."""
    # We assume the center is an XYZ position for this transformation:
    # TODO(kgenova) Update this transformation to account for rotation.

    if self._model_config.hparams.tx == 'i':
      return tf.eye(4, batch_shape=[self.batch_size, self.element_count])

    if 'c' in self._model_config.hparams.tx:
      tx = tf.eye(3, batch_shape=[self.batch_size, self.element_count])
      centers = tf.reshape(self._centers,
                           [self.batch_size, self.element_count, 3, 1])
      tx = tf.concat([tx, -centers], axis=-1)
      lower_row = tf.constant(
          np.tile(
              np.reshape(np.array([0., 0., 0., 1.]), [1, 1, 1, 4]),
              [self.batch_size, self.element_count, 1, 1]),
          dtype=tf.float32)
      tx = tf.concat([tx, lower_row], axis=-2)
    else:
      tx = tf.eye(4, batch_shape=[self.batch_size, self.element_count])

    # Compute a rotation transformation if necessary:
    if ('r' in self._model_config.hparams.tx) and (
        self._model_config.hparams.r == 'cov'):
      # Apply the inverse rotation:
      rotation = tf.matrix_inverse(
          camera_util.roll_pitch_yaw_to_rotation_matrices(self._radii[...,
                                                                      3:6]))
    else:
      rotation = tf.eye(3, batch_shape=[self.batch_size, self.element_count])

    # Compute a scale transformation if necessary:
    if ('s' in self._model_config.hparams.tx) and (
        self._model_config.hparams.r in ['aa', 'cov']):
      diag = self._radii[..., 0:3]
      diag = 1.0 / (tf.sqrt(diag + 1e-8) + 1e-8)
      scale = tf.matrix_diag(diag)
    else:
      scale = tf.eye(3, batch_shape=[self.batch_size, self.element_count])

    # Apply both transformations and return the transformed points.
    tx3x3 = tf.matmul(scale, rotation)
    return tf.matmul(homogenize(tx3x3), tx)

  def implicit_values(self, local_samples):
    """Computes the implicit values given local input locations.

    Args:
      local_samples: Tensor with shape [..., effective_element_count,
        sample_count, 3]. The samples, which should already be in the coordinate
        frame of the local coordinates.

    Returns:
      values: Tensor with shape [..., effective_element_count, sample_count, 1]
        or something (like a scalar) that can broadcast to that type. The value
        decoded from the implicit parameters at each element.
    """
    if self._model_config.hparams.ipe == 'f':
      log.warning(
          "Requesting implicit values when ipe='f'. iparams are None? %s" %
          repr(self._iparams is None))
      raise ValueError(
          "Can't request implicit values when ipe='f'.")
    elif self._model_config.hparams.ipe not in ['e', 't']:
      raise ValueError('Unrecognized ipe hparam: %s' %
                       self._model_config.hparams.ipe)
    else:
      iparams = _tile_for_symgroups(self._model_config, self._iparams)
      eec = iparams.get_shape().as_list()[-2]
      sample_eec = local_samples.get_shape().as_list()[-3]
      if eec != sample_eec:
        raise ValueError(
            'iparams have element count %i, local samples have element_count %i'
            % (eec, sample_eec))
      values = self.net.eval_implicit_parameters(iparams, local_samples)
      # TODO(kgenova) Maybe this should be owned by a different piece of code?
      if self._model_config.hparams.ipe == 'e':
        values = tf.sigmoid(values)  # The parameters define a decision.
      return values

  def rbf_influence_at_samples(self, samples):
    """Computes the per-effective-element RBF weights at the input samples.

    Args:
      samples: Tensor with shape [..., sample_count, 3]. The input samples.

    Returns:
      Tensor with shape [..., sample_count, effective_element_count]. The RBF
        weight of each *effective* element at each position. The effective
        elements are determined by the SIF's symmetry groups.
    """
    batching_dims = samples.get_shape().as_list()[:-2]
    batching_rank = len(batching_dims)
    # For now:
    assert batching_rank == 1
    sample_count = samples.get_shape().as_list()[-2]

    effective_constants = _tile_for_symgroups(self._model_config,
                                              self._constants)
    effective_centers = _tile_for_symgroups(self._model_config, self._centers)
    effective_radii = _tile_for_symgroups(self._model_config, self._radii)
    # Gives the samples shape [batch_size, effective_elt_count, sample_count, 3]
    effective_samples = _generate_symgroup_samples(self._model_config, samples)
    effective_element_count = get_effective_element_count(self._model_config)

    _, per_element_weights = (
        quadrics.compute_shape_element_influences(
            constants_to_quadrics(effective_constants), effective_centers,
            effective_radii, effective_samples))
    weights = tf.ensure_shape(
        per_element_weights,
        batching_dims + [effective_element_count, sample_count, 1])
    weights = tf.reshape(
        weights, batching_dims + [effective_element_count, sample_count])

    # To get to the desired output shape we need to swap the final dimensions:
    perm = list(range(len(weights.get_shape().as_list())))
    perm[-1], perm[-2] = perm[-2], perm[-1]
    assert perm[-2] > perm[-1]
    weights = tf.transpose(weights, perm=perm)
    return weights

  def class_at_samples(self, samples, apply_class_transfer=True):
    """Computes the function value of the implicit function at input samples.

    Args:
      samples: Tensor with shape [..., sample_count, 3]. The input samples.
      apply_class_transfer: Whether to apply a class transfer function to the
        predicted values. If false, will be the algebraic distance (depending on
        the selected reconstruction equations).

    Returns:
      A tuple: (global_decisions, local_information).
        global_decisions: Tensor with shape [..., sample_count, 1]. The
          classification value at each sample from the overall reconstruction.
        local_information: A tuple with two entries:
          local_decisions: A [..., element_count, sample_count, 1] Tensor. The
            output value at each sample from the individual shape elements. This
            value may not always be interpretable as a classification. If the
            global solution is an interpolation of local solutions, it will be.
            Otherwise, it may only be interpretable as a marginal contribution
            to the global classification decision.
          local_weights: A [..., element_count, sample_count, 1] Tensor. The
            influence weights of the local decisions.
    """
    batching_dims = samples.get_shape().as_list()[:-2]
    batching_rank = len(batching_dims)
    # For now:
    assert batching_rank == 1
    # assert batching_rank in [0, 1]
    # if batching_rank == 0:
    #   batching_rank = 1
    #   batching_dims = [1]
    sample_count = samples.get_shape().as_list()[-2]

    effective_constants = _tile_for_symgroups(self._model_config,
                                              self._constants)
    effective_centers = _tile_for_symgroups(self._model_config, self._centers)
    effective_radii = _tile_for_symgroups(self._model_config, self._radii)

    effective_samples = _generate_symgroup_samples(self._model_config, samples)
    # The samples have shape [batch_size, effective_elt_count, sample_count, 3]
    effective_element_count = get_effective_element_count(self._model_config)

    per_element_constants, per_element_weights = (
        quadrics.compute_shape_element_influences(
            constants_to_quadrics(effective_constants), effective_centers,
            effective_radii, effective_samples))
    per_element_constants = tf.ensure_shape(
        per_element_constants,
        batching_dims + [effective_element_count, sample_count, 1])
    per_element_weights = tf.ensure_shape(
        per_element_weights,
        batching_dims + [effective_element_count, sample_count, 1])

    agg_fun_dict = {
        's': tf.reduce_sum,
        'm': tf.reduce_max,
        'l': tf.reduce_logsumexp,
        'v': tf.compat.v1.math.reduce_variance,
    }
    agg_fun = agg_fun_dict[self._model_config.hparams.ag]

    if self._model_config.hparams.ipe == 'f':
      local_decisions = per_element_constants * per_element_weights
      local_weights = per_element_weights
      sdf = agg_fun(local_decisions, axis=batching_rank)

    # We currently have constants, weights with shape:
    # [batch_size, element_count, sample_count, 1].
    # We need to use the net to get a same-size grid of offsets.
    # The input samples to the net must have shape
    # [batch_size, element_count, sample_count, 3], while the current samples
    # have shape [batch_size, sample_count, 3]. This is because each sample
    # should be evaluated in the relative coordinate system of the
    if self._model_config.hparams.ipe in ['t', 'e']:
      effective_world2local = _tile_for_symgroups(self._model_config,
                                                  self.world2local)
      local_samples = _transform_samples(effective_samples,
                                         effective_world2local)

      implicit_values = self.implicit_values(local_samples)

    if self._model_config.hparams.ipe == 't':
      multiplier = 1.0 if self._model_config.hparams.ipc == 't' else 0.0
      residuals = 1 + multiplier * implicit_values
      # Each element is c * w * (1 + OccNet(x')):
      local_decisions = per_element_constants * per_element_weights * residuals
      local_weights = per_element_weights
      sdf = agg_fun(local_decisions, axis=batching_rank)

    if self._model_config.hparams.ipe in ['f', 't']:
      # Need to map from the metaball influence to something that's < 0 inside.
      if apply_class_transfer:
        sdf = sdf_util.apply_class_transfer(
            sdf,
            self._model_config,
            soft_transfer=True,
            offset=self._model_config.hparams.lset)

    if self._model_config.hparams.ipc != 't':
      return sdf, (local_decisions, local_weights)

    if self._model_config.hparams.ipe == 'e':
      local_decisions = implicit_values
      weighted_constants = per_element_weights * tf.abs(per_element_constants)
      occnet_weights = tf.nn.softmax(weighted_constants, axis=batching_rank)
      sdf_components = occnet_weights * local_decisions
      local_weights = occnet_weights
      # This is the class at the samples. It is a probability of being outside.
      sdf = agg_fun(sdf_components, axis=batching_rank)

    sdf = tf.ensure_shape(sdf, batching_dims + [sample_count, 1])
    local_decisions = tf.ensure_shape(
        local_decisions,
        batching_dims + [effective_element_count, sample_count, 1])
    # TODO(kgenova) Maybe we should plot the histogram of values...
    return sdf, (local_decisions, local_weights)

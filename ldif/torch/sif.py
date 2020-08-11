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
"""Minimal support for SIF evaluation in pytorch."""

import numpy as np

import torch

from ldif.util import file_util
from ldif.util.file_util import log

import sif_evaluation


def ensure_are_tensors(named_tensors):
  for name, tensor in named_tensors.items():
    if not (torch.is_tensor(tensor)):
      raise ValueError(f'Argument {name} is not a tensor, it is a {type(tensor)}')

def ensure_type(v, t, name):
  if not isinstance(v, t):
    raise ValueError(f'Error: variable {name} has type {type(v)}, not the expected type {t}')

def _load_v1_txt(path):
  """Parses a SIF V1 text file, returning numpy arrays.
  
  Args:
    path: string containing the path to the ASCII file.
    
  Returns:
    A tuple of 4 elements:
      constants: A numpy array of shape (element_count). The constant
        associated with each SIF element.
      centers: A numpy array of shape (element_count, 3). The centers of the
        SIF elements.
      radii: A numpy array of shape (element_count, 3). The axis-aligned
        radii of the gaussian falloffs.
      rotations: A numpy array of shape (element_count, 3). The euler-angle
        rotations of the SIF elements.
      symmetry_count: An integer. The number of elements which are left-right
        symmetric.
      features: A numpy array of shape (element_count, implicit_len). The LDIF
        neural features, if they are present. 
  """
  lines = file_util.readlines(path)
  if lines[0] != 'SIF':
    raise ValueError(f'Could not parse {path} as a sif txt. First line was {lines[0]}')
  shape_count, version, implicit_len = [int(x) for x in lines[1].split(' ')]
  version += 1
  if version != 1:
    raise ValueError(f'This function can only parse v1 files. This version: {version}.')
  symmetry_count = 0
  last_was_symmetric = True
  constants = []
  centers = []
  radii = []
  rotations = []
  features = []
  for row in lines[2:]:
    elts = row.split(' ')
    if len(elts) != 11 + implicit_len:
      raise ValueError('Failed to parse the following row with '
          f'implicit_len {implicit_len}: {row}')
    explicit_params = np.array([float(x) for x in elts[:10]], dtype=np.float32)
    is_symmetric = bool(int(elts[10]))
    if is_symmetric:
      symmetry_count += 1
      if not last_was_symmetric:
        raise ValueError(f'File not supported by parser: row {row} is '
            'symmetric but follows an asymmetric element.')
    constants.append(explicit_params[0])
    centers.append(explicit_params[1:4])
    radii.append(explicit_params[4:7])
    rotations.append(explicit_params[7:10])
    if implicit_len > 0:
      implicit_params = np.array([float(x) for x in elts[11:]], dtype=np.float32)
      features.append(implicit_params)
  constants = np.stack(constants)
  centers = np.stack(centers)
  radii = np.stack(radii)
  rotations = np.stack(rotations)
  features = np.stack(features) if features else None
  # Radii have their sqrt stored for GAPS:
  radii = radii * radii
  return constants, centers, radii, rotations, symmetry_count, features


def _tile_for_symgroups(elements, symmetry_count):
  """Tiles an input tensor along its element dimension based on symmetry.

  Args:
    elements: Tensor with shape [batch_size, element_count, ...].

  Returns:
    Tensor with shape [batch_size, element_count + tile_count, ...]. The
    elements have been tiled according to the model configuration's symmetry
    group description.
  """
  left_right_sym_count = symmetry_count
  assert len(elements.shape) >= 3
  # The first K elements get reflected with left-right symmetry (z-axis) as
  # needed.
  if left_right_sym_count:
    first_k = elements[:, :left_right_sym_count, ...]
    elements = torch.cat([elements, first_k], axis=1)
  # TODO(kgenova) As additional symmetry groups are added, add their tiling.
  return elements


def reflect_z(samples):
  """Reflects the sample locations across the planes specified in xyz.

  Args:
    samples: Tensor with shape [..., 3].

  Returns:
    Tensor with shape [..., 3]. The reflected samples.
  """
  to_keep = samples[..., :-1]
  to_reflect = samples[..., -1:]
  return torch.cat([to_keep, -to_reflect], dim=-1)


def _generate_symgroup_samples(samples, element_count, symmetry_count):
  """Duplicates and transforms samples as needed for symgroup queries.

  Args:
    samples: Tensor with shape [batch_size, sample_count, 3].

  Returns:
    Tensor with shape [batch_size, effective_element_count, sample_count, 3].
  """
  if len(samples.shape) != 3:
    raise ValueError(f'Internal Error: Samples have shape {samples.shape}')
  bs, sample_count = samples.shape[:2]
  samples = torch.reshape(samples, [bs, 1, sample_count, 3]).expand([-1, element_count, -1, -1])

  left_right_sym_count = symmetry_count
  if left_right_sym_count:
    first_k = samples[:, :left_right_sym_count, :, :]
    first_k = reflect_z(first_k)
    samples = torch.cat([samples, first_k], axis=1)
  return samples

def _generate_symgroup_frames(world2local, symmetry_count):
  """Duplicates and adds reflection transformation for symgroup matrices.

  Args:
    world2local: Tensor with shape [bs, element_count, 4, 4].
    symmetry_count: Int, at most element_count. The number of LR symmetric frames.

  Returns:
    Tensor with shape [bs, effective_element_count, 4, 4].
  """
  if len(world2local.shape) != 4:
    raise ValueError(f'Invalid world2local shape: {world2local.shape}')
  if symmetry_count:
    bs = world2local.shape[0]
    reflector = torch.eye(4).cuda()
    first_two_rows = reflector[:2, :]
    last_row = reflector[3:, :]
    reflected = -1 * reflector[2:3, :]
    reflector = torch.cat([first_two_rows, reflected, last_row], dim=0)
    reflector = torch.reshape(reflector, [1, 1, 4, 4]).expand([bs, symmetry_count,
        -1, -1])
    first_k = world2local[:, :symmetry_count, :, :]
    first_k = torch.matmul(first_k, reflector)
    world2local = torch.cat([world2local, first_k], axis=1)
  return world2local



class Sif(object):
  """A SIF for loading from txts, packing into a tensor, and evaluation."""

  def __init__(self, constants, centers, radii, rotations, symmetry_count):
    """The real initializer (from tensors). Not intended for direct use (see below)."""
    ensure_are_tensors({'constants': constants, 'centers': centers,
        'radii': radii, 'rotations': rotations})
    if not (isinstance(symmetry_count, int)):
      raise ValueError(f'symmetry_count is of type: {type(symmetry_count)}.')
    has_batch_dim = len(centers.shape) == 3
    if not has_batch_dim and len(centers.shape) != 2:
      raise ValueError(f'Unable to parse input tensor shape: {centers.shape}')
    bs = centers.shape[0] if has_batch_dim else 1
    element_count = centers.shape[-2]
    assert isinstance(element_count, int)
    self._constants = torch.reshape(constants, (bs, element_count, 1))
    self._centers = torch.reshape(centers, (bs, element_count, 3))
    self._radii = torch.reshape(radii, (bs, element_count, 3))
    self._rotations = torch.reshape(rotations, (bs, element_count, 3))
    self.bs = bs
    self.element_count = element_count
    self.symmetry_count = symmetry_count

  @classmethod
  def from_file(cls, path):
    """Generates a SIF object from one or more txt files.
    
    Args:
      path: Either a string containing the path to a txt file, or a list of
        one or more strings, each containing a path to a text file.

    Returns:
      A SIF object. It will have a batch dimension containing each of the
      txts in order, or no batch dimension if only one path was provided.
    """
    if len(path) == 1 or isinstance(path, str):
      loaded = _load_v1_txt(path)
      symmetry_count, features = loaded[-2:]
      explicits = [torch.Tensor(x).cuda() for x in loaded[:-2]]
      # TODO(kgenova) Consider adding support to ignore the features, rather
      # than just crashing. It would be fine to just throw them away.
      if features is not None:
        raise ValueError(f'This class cannot handle LDIFs, only SIFs.')
      return cls(*explicits, symmetry_count)
    flattened_shapes = []
    symc = None
    ec = None
    for p in path:
      shape = cls.from_file(p)
      flat, cur_symc = shape.to_flat_tensor()
      flattened_shapes.append(flat)
      if symc is None:
        symc = cur_symc
      if symc != cur_symc:
        raise ValueError('Trying to make a batched SIF with mismatched '
          f'symmetry: {symc} vs {cur_symc}')
      if ec is None:
        ec = shape.element_count
      if ec != shape.element_count:
        raise ValueError('Trying to make a batched SIF with mismatched '
          f'element counts: {ec} vs {shape.element_count}')
    return cls.from_flat_tensor(torch.cat(flattened_shapes), symc)

  @classmethod
  def from_flat_tensor(cls, tensor, symmetry_count):
    """Generates a batch of SIFs from a batched tensor.

    The symmetry_count is an integer because for batched SIFs, it is required
    that all SIFs share the same symmetry count.

    Args:
      tensor: A single tensor previously generated by to_flat_tensor.
      symmetry_count: The symmetry count variable for the SIFs (an int).

    Returns:
      A Sif object that can evaluate a batch of SIFs at once.
    """
    if not torch.is_tensor(tensor):
      raise ValueError(f'Input is not a tensor, but a {type(tensor)}: {tensor}')
    if len(tensor.shape) != 3 or tensor.shape[-1] != 10:
      raise ValueError(f'Could not parse flat tensor due to shape: {tensor.shape}')
    constants = tensor[:, :, :1]
    centers = tensor[:, :, 1:4]
    radii = tensor[:, :, 4:7]
    rotations = tensor[:, :, 7:10]
    return cls(constants, centers, radii, rotations, symmetry_count)
     

  def to_flat_tensor(self):
    """Generates a single tensor from the SIF. Can be batched with torch.cat.

    Works with either single or batch SIFs. Useful for preloading the SIFs
    so loading is not a bottleneck during training.
    
    Returns:
      1) A tensor with shape [bs, element_count, 10]
      2) An int. The symmetry count, needed to restore the SIFs. Note that
        this must be the same for all SIFs in a batch, which is why only
        a single int is returned. 
    """
    flat = torch.cat((self._constants, self._centers, self._radii, self._rotations),
        dim=2)
    return flat, self.symmetry_count
  
  def rbf_influence(self, samples):
    """Evaluates the influence of each RBF in the SIF at each sample.

    Args:
      samples: A tensor containing the samples, in the SIF global frame.
        Has shape (sample_count, 3) or (bs, sample_count_3).

    Returns:
      A tensor with shape (sample_count, effective_element_count) or
      (bs, sample_count, effective_element_count). The 'effective' element
      count may be higher than the element count, depending on the symmetry
      settings of the SIF. In the case were the SIF is at least partially
      symmetric, then some elements have multiple RBF weights- their main
      weight (given first) and the weight associated with the 'shadow'
      element(s) transformed by their symmetry matrix. See get_symmetry_map()
      for a mapping from original element indices to their symmetric 
      counterparts. Regardless of additional 'shadow' elements, the first 
      element_count RBF weights correspond to the 'real' elements with no
      symmetry transforms applied, in order.
    """
    if len(samples.shape) == 2:
      if self.bs != 1:
        raise ValueError('Samples must have a batch dimension if the SIF does.'
            f' Input sample shape was {samples.shape} and SIF bs is {self.bs}')
      samples = torch.unsqueeze(samples, dim=0)

    samples = _generate_symgroup_samples(samples, self.element_count,
      self.symmetry_count)
    weights = sif_evaluation.compute_rbf_influences(
      self._tiled_centers, self._tiled_radii, self._tiled_rotations, samples)
    assert len(weights.shape) == 4
    assert weights.shape[-1] == 1
    # Currently last dim is always 1 and it's [bs, eec, sc, 1], not
    # [bs, sc, eec] or [sc, eec] as needed
    weights = weights[..., 0]
    weights = torch.transpose(weights, -2, -1)
    if self.bs == 1:
      return weights[0, ...]
    return weights

  @property
  def effective_element_count(self):
    """The number of elements, accounting for symmetry."""
    return self.element_count + self.symmetry_count

  @property
  def constants(self):
    """The constant parameters associated with the SIF.

    Returns:
      A tensor with shape (effective_element_count) or
      (bs, effective_element_count). See rbf_influence for an explanation
      of how to 'effective' samples.
    """
    if self.bs == 1:
      return self._tiled_constants[0, :, 0]
    return self._tiled_constants[:, :, 0]

  @property
  def _tiled_constants(self):
    """The constants, tiled to account for symmetry."""
    if not hasattr(self, '__tiled_constants'):
      self.__tiled_constants = _tile_for_symgroups(self._constants, self.symmetry_count)
    return self.__tiled_constants
  
  @property
  def _tiled_centers(self):
    """The centers, tiled to account for symmetry."""
    if not hasattr(self, '__tiled_centers'):
      self.__tiled_centers = _tile_for_symgroups(self._centers, self.symmetry_count)
    return self.__tiled_centers

  @property
  def _tiled_radii(self):
    """The radii, tiled to account for symmetry."""
    if not hasattr(self, '__tiled_radii'):
      self.__tiled_radii = _tile_for_symgroups(self._radii, self.symmetry_count)
    return self.__tiled_radii

  @property
  def _tiled_rotations(self):
    """The rotations, tiled to account for symmetry."""
    if not hasattr(self, '__tiled_rotations'):
      self.__tiled_rotations = _tile_for_symgroups(self._rotations, self.symmetry_count)
    return self.__tiled_rotations

  @property
  def world2local(self):
    """The 4x4 transformation matrices associated with the SIF elements.

    Returns:
      A tensor of shape (effective_element_count, 4, 4) or 
      (bs, effective_element_count, 4, 4). See rbf_influence for an explanation
      of element_count vs effective_element_count.
    """
    if not hasattr(self, '_world2local'):
      self._world2local = sif_evaluation.compute_world2local(self._centers,
          self._radii, self._rotations)
      self._world2local = _generate_symgroup_frames(self._world2local,
          self.symmetry_count)
      if self.bs == 1:
        self._world2local = torch.reshape(self._world2local,
            [self.effective_element_count, 4, 4])
    return self._world2local

  def eval(self, samples):
    """Evaluates the SIF at the samples.

    Args:
      samples: A tensor of shape (sample_count, 3) or (bs, sample_count, 3).
        The locations to evaluate the SIF, in the SIF's world coordinate frame.

    Returns:
      A tensor of shape (sample_count) or (bs, sample_count). The value of the
        SIF at each sample point. Typically, values less than -0.07 are inside
        and values greater than -0.07 are outside.
    """
    # TODO(kgenova) A future version of the SIF txt file should contain the
    # isosurface used for inside/outside determination, so users don't have
    # to keep that information around.
    if self.bs > 1 and len(samples.shape) == 2:
      samples = torch.unsqueeze(samples, dim=0).expand([self.bs, -1, -1])
    cs = self.constants
    sample_count = samples.shape[-1]
    cs = torch.unsqueeze(cs, dim=-2)
    rbfs = self.rbf_influence(samples)
    result = cs * rbfs
    return torch.sum(result, dim=-1)



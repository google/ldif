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


def ensure_are_tensors(named_tensors):
  for name, tensor in named_tensors.items():
    if not (torch.is_tensor(tensor)):
      raise ValueError(f'Argument {name} is not a tensor, it is a {type(tensor)}')

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
    self._constants = torch.reshape(constants, (bs, element_count, 1))
    self._centers = torch.reshape(centers, (bs, element_count, 3))
    self._radii = torch.reshape(radii, (bs, element_count, 3))
    self._rotations = torch.reshape(rotations, (bs, element_count, 3))
    self.bs = bs
    self.element_count = element_count
    self.symmetry_count = symmetry_count

  @classmethod
  def from_file(cls, path):
    """Generates a SIF object from a txt file."""
    loaded = _load_v1_txt(path)
    symmetry_count, features = loaded[-2:]
    explicits = [torch.Tensor(x).cuda() for x in loaded[:-2]]
    # TODO(kgenova) Consider adding support to ignore the features, rather
    # than just crashing. It would be fine to just throw them away.
    if features is not None:
      raise ValueError(f'This class cannot handle LDIFs, only SIFs.')
    return cls(*explicits, symmetry_count)

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
    if len(tensor.shape) != 3 or tensor.shape[-1] != 10:
      raise ValueError(f'Could not parse flat tensor due to shape: {tensor.shape}')
    constants = tensor[:, :, :1]
    centers = tensor[:, :, 1:4]
    radii = tensor[:, :, 4:7]
    rotations = tensor[:, :, 7:10]
    return cls(constants, centers, radii, rotations, symmetry_count)
     

  def to_flat_tensor(self):
    """Generates a single tensor from the SIF. Can be batched with torch.cat.

    Works with either single or batch SIFs.
    
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
    pass

  def constants(self):
    """The constant parameters associated with the SIF.

    Returns:
      A tensor with shape (effective_element_count) or
      (bs, effective_element_count). See rbf_influence for an explanation
      of how to 'effective' samples.
    """
    if self.bs == 1:
      return self._constants[0, :]
    return self._constants[:, :]

  def world2local(self):
    """The 4x4 transformation matrices associated with the SIF elements.

    Returns:
      A tensor of shape (effective_element_count, 4, 4) or 
      (bs, effective_element_count, 4, 4). See rbf_influence for an explanation
      of element_count vs effective_element_count.
    """
    pass

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
    pass



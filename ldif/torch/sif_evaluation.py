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
"""Internals for evaluating a Sif with PyTorch."""

import torch

DIV_EPSILON = 1e-8

def ensure_shape(tensor, shape):
  msg = f'Shape Mismatch: found {tensor.shape} vs expected {shape}'
  if len(tensor.shape) != len(shape):
    raise ValueError(msg)
  for i in range(len(shape)):
    if shape[i] != -1 and tensor.shape[i] != shape[i]:
      raise ValueError(msg)


def roll_pitch_yaw_to_rotation_matrices(roll_pitch_yaw):
  """Converts roll-pitch-yaw angles to rotation matrices.

  Args:
    roll_pitch_yaw: Tensor with shape [..., 3]. The last dimension contains
      the roll, pitch, and yaw angles in radians.  The resulting matrix
      rotates points by first applying roll around the x-axis, then pitch 
      around the y-axis, then yaw around the z-axis.

  Returns:
     Tensor with shape [..., 3, 3]. The 3x3 rotation matrices corresponding to
     the input roll-pitch-yaw angles.
  """

  cosines = torch.cos(roll_pitch_yaw)
  sines = torch.sin(roll_pitch_yaw)
  cx, cy, cz = torch.unbind(cosines, dim=-1)
  sx, sy, sz = torch.unbind(sines, dim=-1)
  # pyformat: disable
  rotation = torch.stack(
      [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
       sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
       -sy, cy * sx, cy * cx], dim=-1)
  # pyformat: enable
  #shape = torch.cat([roll_pitch_yaw.shape[:-1], [3, 3]], axis=0)
  shape = list(roll_pitch_yaw.shape[:-1]) + [3, 3]
  rotation = torch.reshape(rotation, shape)
  return rotation


def decode_covariance_roll_pitch_yaw(radii, rotations, invert=False):
  """Converts 6-D radus vectors to the corresponding covariance matrices.
  Args:
    radii: Tensor with shape [..., 3]. Covariances of the three Gaussian axes. 
    rotations: Tensor with shape [..., 3]. The roll-pitch-yaw rotation angles
      of the Gaussian frame.
    invert: Whether to return the inverse covariance.
  Returns:
     Tensor with shape [..., 3, 3]. The 3x3 (optionally inverted) covariance
     matrices corresponding to the input radius vectors.
  """
  d = 1.0 / (radii + DIV_EPSILON) if invert else radii
  diag = torch.diag_embed(d)
  rotation = roll_pitch_yaw_to_rotation_matrices(rotations)
  return torch.matmul(torch.matmul(rotation, diag), torch.transpose(rotation, -2, -1))


def sample_cov_bf(center, radii, rotations, samples):
  """Samples gaussian radial basis functions at specified coordinates.
  Args:
    center: Tensor with shape [..., 3]. Contains the [x,y,z] coordinates of the
      RBF center in NIC space with a top-left origin.
    radii: Tensor with shape [..., 3]. Covariances of the three Gaussian axes. 
    rotations: Tensor with shape [..., 3]. The roll-pitch-yaw rotation angles
      of the Gaussian frame.
    samples: Tensor with shape [..., N, 3],  where N is the number of samples to
      evaluate. These are the sample locations in the same frame in which the
      center is defined. Supports broadcasting the batching dimensions.
  Returns:
     Tensor with shape [..., N, 1]. The basis function strength at each sample
     location.
  """
  # Compute the samples' offset from center, then extract the coordinates.
  diff = samples - torch.unsqueeze(center, dim=-2)
  x, y, z = torch.unbind(diff, dim=-1)
  # Decode 6D radius vectors into inverse covariance matrices, then extract
  # unique elements.
  inv_cov = decode_covariance_roll_pitch_yaw(radii, rotations, invert=True)
  shape = list(inv_cov.shape[:-2]) + [1, 9]
  inv_cov = torch.reshape(inv_cov, shape)
  c00, c01, c02, _, c11, c12, _, _, c22 = torch.unbind(inv_cov, dim=-1)
  # Compute function value.
  dist = (
      x * (c00 * x + c01 * y + c02 * z) + y * (c01 * x + c11 * y + c12 * z) +
      z * (c02 * x + c12 * y + c22 * z))
  dist = torch.exp(-0.5 * dist)
  return dist


def compute_rbf_influences(centers, radii, rotations, samples):
  """Computes the per-shape-element RBF values at given sample locations.
  Args:
    radii: rbf radii with shape [batch_size, element_count, 3].
    rotations: euler-angle rotations with shape [batch_size, element_count, 3].
    samples: a grid of samples with shape [batch_size, element_count,
      sample_count, 3] or shape [batch_size, sample_count, 3].
  Returns:
    Tensor with shape [batch_size, element_count, sample_count, 1]
  """
  # Select the number of samples along the ray. The larger this is, the
  # more memory that will be consumed and the slower the algorithm. But it
  # reduces warping artifacts and the likelihood of missing a thin surface.
  batch_size, element_count = centers.shape[:2]
  ensure_shape(centers, [batch_size, element_count, 3])
  ensure_shape(radii, [batch_size, element_count, 3])
  ensure_shape(rotations, [batch_size, element_count, 3])
  
  # Ensure the samples have the right shape and tile in an axis for the
  # quadric dimension if it wasn't provided.
  sample_rank = len(samples.shape)
  if (sample_rank not in [3, 4] or samples.shape[-1] != 3 or
      samples.shape[0] != batch_size):
      raise ValueError(
          'Input tensor samples must have shape [batch_size, element_count,'
          ' sample_count, 3] or shape [batch_size, sample_count, 3]. The input'
          f' shape was {samples.shape}')
  missing_element_dim = sample_rank == 3
  sample_count = samples.shape[-2]
  if missing_element_dim:
      samples = torch.reshape(samples, [batch_size, 1, sample_count, 3]).expand(
          [-1, element_count, -1, -1])
  ensure_shape(samples, [batch_size, element_count, sample_count, 3])
  
  sampled_rbfs = sample_cov_bf(centers, radii, rotations, samples)
  sampled_rbfs = torch.reshape(sampled_rbfs,
                              [batch_size, element_count, sample_count, 1])
  return sampled_rbfs

def homogenize(m):
  """Adds homogeneous coordinates to a [..., N,N] matrix, returning [..., N+1, N+1]."""
  assert m.shape[-1] == m.shape[-2]  # Must be square
  n = m.shape[-1]
  eye_n_plus_1 = torch.eye(n+1).cuda().expand(list(m.shape[:-2]) + [-1, -1])
  extra_col = eye_n_plus_1[..., :-1, -1:]
  extra_row = eye_n_plus_1[..., -1:, :]
  including_col = torch.cat([m, extra_col], dim=-1)
  return torch.cat([including_col, extra_row], dim=-2)


def compute_world2local(centers, radii, rotations):
  """Computes a transformation to the local element frames for encoding."""
  # We assume the center is an XYZ position for this transformation:
  # TODO(kgenova) Update this transformation to account for rotation.
  assert len(centers.shape) == 3
  batch_size, element_count = centers.shape[:2]
  
  eye_3x3 = torch.eye(3).cuda().expand([batch_size, element_count, -1, -1])
  eye_4x4 = torch.eye(4).cuda().expand([batch_size, element_count, -1, -1])
  
  # Centering transform
  # ones = torch.ones([batch_size, element_count, 1, 1])
  centers = torch.reshape(centers,
                      [batch_size, element_count, 3, 1])
  tx = torch.cat([eye_3x3, -centers], dim=-1)
  tx = torch.cat([tx, eye_4x4[..., 3:4, :]], dim=-2)  # Append last row
  
  # Compute the inverse rotation:
  rotation = torch.inverse(roll_pitch_yaw_to_rotation_matrices(rotations))
  assert rotation.shape == (batch_size, element_count, 3, 3)
  
  # Compute a scale transformation:
  diag = 1.0 / (torch.sqrt(radii + 1e-8) + 1e-8)
  scale = torch.diag_embed(diag)
  
  # Apply both transformations and return the transformed points.
  tx3x3 = torch.matmul(scale, rotation)
  return torch.matmul(homogenize(tx3x3), tx)

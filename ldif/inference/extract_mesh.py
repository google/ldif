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
"""Converts a structured implicit function into a mesh."""

import numpy as np
from skimage import measure
import trimesh
# ldif is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order


def marching_cubes(volume, mcubes_extent):
  """Maps from a voxel grid of implicit surface samples to a Trimesh mesh."""
  volume = np.squeeze(volume)
  length, height, width = volume.shape
  resolution = length
  # This function doesn't support non-cube volumes:
  assert resolution == height and resolution == width
  thresh = -0.07
  try:
    vertices, faces, normals, _ = measure.marching_cubes_lewiner(volume, thresh)
    del normals
    x, y, z = [np.array(x) for x in zip(*vertices)]
    xyzw = np.stack([x, y, z, np.ones_like(x)], axis=1)
    # Center the volume around the origin:
    xyzw += np.array(
        [[-resolution / 2.0, -resolution / 2.0, -resolution / 2.0, 0.]])
    # This assumes the world is right handed with y up; matplotlib's renderer
    # has z up and is left handed:
    # Reflect across z, rotate about x, and rescale to [-0.5, 0.5].
    xyzw *= np.array([[(2.0 * mcubes_extent) / resolution,
                       (2.0 * mcubes_extent) / resolution,
                       -1.0 * (2.0 * mcubes_extent) / resolution, 1]])
    y_up_to_z_up = np.array([[0., 0., -1., 0.], [0., 1., 0., 0.],
                             [1., 0., 0., 0.], [0., 0., 0., 1.]])
    xyzw = np.matmul(y_up_to_z_up, xyzw.T).T
    faces = np.stack([faces[..., 0], faces[..., 2], faces[..., 1]], axis=-1)
    world_space_xyz = np.copy(xyzw[:, :3])
    mesh = trimesh.Trimesh(vertices=world_space_xyz, faces=faces)
    log.verbose('Generated mesh successfully.')
    return True, mesh
  except (ValueError, RuntimeError) as e:
    log.warning(
        'Failed to extract mesh with error %s. Setting to unit sphere.' %
        repr(e))
    return False, trimesh.primitives.Sphere(radius=0.5)

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
"""Numpy interface for the ShapeNet dataset."""

import numpy as np


class ShapeNetExampleNp(object):
  """A single ShapeNet training example."""

  def __init__(self, proto):
    self._proto = proto
    self._depth_renders = None

  @property
  def depth_renders(self):
    if self._depth_renders is None:
      depth_renders = self._proto.depth_renders
      depth_renders = depth_renders.astype(np.float32)
      self._depth_renders = depth_renders / 1000.0  # Was in 1000-ths.
    return self._depth_renders

  @property
  def mesh_renders(self):
    return self._proto.mesh_renders

  @property
  def grid(self):
    return self._proto.grid

  @property
  def world2grid(self):
    return self._proto.world2grid

  @property
  def mesh_name(self):
    return self._proto.mesh_name

  @property
  def bounding_box_samples(self):
    return self._proto.bounding_box_samples

  @property
  def near_surface_samples(self):
    return self._proto.near_surface_samples

  @property
  def surface_points(self):
    return self._proto.surface_point_samples[..., :3]

  @property
  def surface_normals(self):
    return self._proto.surface_point_samples[..., 3:]

  @property
  def bounding_box_points(self):
    return self._proto.bounding_box_samples[..., :3]

  @property
  def bounding_box_sdfs(self):
    return self._proto.bounding_box_samples[..., 3:4]

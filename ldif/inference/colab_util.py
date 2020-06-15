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
"""Utility functions exclusively useful in colab notebooks."""

import matplotlib.pyplot as plt
import numpy as np

from tensorflow_graphics.notebooks import mesh_viewer as threejs_viz

# Previously:
# default view_dir was (0.5, 0.5, 0.0)
# bottom view_dir was (-0.5, -0.5, 0.0
# back view_dir was (-0.5, 0.0, 0.0)
# side view_dir was (0.0, 0.0, 0.5)


def trimesh_to_shape(mesh):
  """Converts a trimesh to a shape dict."""
  if not mesh:
    raise ValueError('Cannot convert an empty trimesh.')
  shape = {'vertices': mesh.vertices, 'faces': mesh.faces}
  if mesh.visual.kind == 'vertex' and mesh.visual.vertex_colors.any():
    shape['vertex_colors'] = np.array(
        mesh.visual.vertex_colors[:, :3], dtype=np.float) / 255.0
  return shape


def show(mesh, res=256):
  shape_viewer = threejs_viz.Viewer(mesh)
  del res
  return shape_viewer


def plot(im):
  im = np.squeeze(im)
  plt.imshow(im)
  plt.grid(b=None)
  plt.axis('off')


def plot_all(ims):
  im = np.concatenate(ims, axis=1)
  plot(im)

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
"""Utilities for working with meshes."""

import io

import trimesh


def serialize(mesh):
  mesh_str = trimesh.exchange.ply.export_ply(
      mesh, encoding='binary', vertex_normal=False)
  return mesh_str


def deserialize(mesh_str):
  mesh_ply_file_obj = io.BytesIO(mesh_str)
  mesh = trimesh.Trimesh(**trimesh.exchange.ply.load_ply(mesh_ply_file_obj))
  return mesh


def remove_small_components(mesh, min_volume=5e-05):
  """Removes all components with volume below the specified threshold."""
  if mesh.is_empty:
    return mesh
  out = [m for m in mesh.split(only_watertight=False) if m.volume > min_volume]
  if not out:
    return mesh
  return trimesh.util.concatenate(out)


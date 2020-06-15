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
"""Library code to create an LDIF example directory from a file."""

import subprocess as sp
import numpy as np

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import gaps_util
from ldif.util import path_util
# pylint: enable=g-bad-import-order


def write_depth_and_normals_npz(dirpath, path_out):
  depth_images = gaps_util.read_depth_directory(f'{dirpath}/depth_images', 20)
  normal_images = gaps_util.read_normals_dir(f'{dirpath}/normals', 20)
  depth_images = depth_images[..., np.newaxis]
  arr = np.concatenate([depth_images, normal_images], axis=-1)
  np.savez_compressed(path_out, arr)


def mesh_to_example(codebase_root_dir, mesh_path, dirpath):
  ldif_path = path_util.get_path_to_ldif_root()
  sp.check_output(
      f'{codebase_root_dir}/scripts/process_mesh_local.sh {mesh_path} {dirpath} {ldif_path}',
      shell=True)
  write_depth_and_normals_npz(dirpath, f'{dirpath}/depth_and_normals.npz')

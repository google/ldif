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

import os
import sys
import unittest

# Unit tests need LDIF added to the path, because the scripts aren't
# in the repository top-level directory.
ldif_root = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
sys.path.append(ldif_root)

import numpy as np
import torch

from ldif.torch import sif

TEST_DATA_DIR  = './data'

class TestSif(unittest.TestCase):

  def _ensure_sifs_equal(self, a, b, ith=0):
    end=ith+1
    if ith == -1:
      self.assertEqual(a.bs, b.bs)
      ith = 0
      end = b.bs
    self.assertTrue(torch.all(a._constants.eq(b._constants[ith:end, ...])))
    self.assertTrue(torch.all(a._centers.eq(b._centers[ith:end, ...])))
    self.assertTrue(torch.all(a._radii.eq(b._radii[ith:end, ...])))
    self.assertTrue(torch.all(a._rotations.eq(b._rotations[ith:end, ...])))
    self.assertEqual(a.symmetry_count, b.symmetry_count)
    self.assertEqual(a.element_count, b.element_count)

  def test_load_sif(self):
    sif_path = os.path.join(TEST_DATA_DIR, 'b831f60f211435df5bbc861d0124304c.txt')
    constants, centers, radii, rotations, symmetry_count, features = (
        sif._load_v1_txt(sif_path))
    self.assertEqual(type(symmetry_count), int)
    self.assertEqual(constants.shape, (32,))
    self.assertEqual(centers.shape, (32, 3))
    self.assertEqual(radii.shape, (32, 3))
    self.assertEqual(rotations.shape, (32, 3))
    self.assertEqual(symmetry_count, 16)
    self.assertEqual(features, None)
    self.assertEqual(constants.dtype, np.float32)
    self.assertEqual(centers.dtype, np.float32)
    self.assertEqual(radii.dtype, np.float32)
    self.assertEqual(rotations.dtype, np.float32)
    self.assertAlmostEqual(constants[4], -0.028, places=3)
    self.assertAlmostEqual(radii[4, 1], 0.000352, places=6)
    self.assertAlmostEqual(rotations[4, 2], -0.0532, places=4)

  def test_build_sif(self):
    sif_path = os.path.join(TEST_DATA_DIR, 'b831f60f211435df5bbc861d0124304c.txt')
    shape = sif.Sif.from_file(sif_path)
    self.assertAlmostEqual(shape._constants[0, 4, 0].cpu().numpy(), -0.028, places=3)
    self.assertAlmostEqual(shape._radii[0, 4, 1].cpu().numpy(), 0.000352, places=6)
    self.assertAlmostEqual(shape._rotations[0, 4, 2].cpu().numpy(), -0.0532, places=4)

  def test_flatten_unflatten(self):
    sif_path = os.path.join(TEST_DATA_DIR, 'b831f60f211435df5bbc861d0124304c.txt')
    shape = sif.Sif.from_file(sif_path)
    flattened, symc = shape.to_flat_tensor()
    unflattened = sif.Sif.from_flat_tensor(flattened, symc)
    self._ensure_sifs_equal(shape, unflattened)

  def test_batching(self):
    shapes = []
    flattened_shapes = []
    test_names = ['b831f60f211435df5bbc861d0124304c',
        'b7eefc4c25dd9e49238581dd5a8af82c']
    for name in test_names:
      path = os.path.join(TEST_DATA_DIR, name + '.txt')
      shape = sif.Sif.from_file(path)
      flattened_shapes.append(shape.to_flat_tensor())
    flats = [x[0] for x in flattened_shapes]
    symcs = [x[1] for x in flattened_shapes]

    for symc in symcs:
      self.assertEqual(symc, symcs[0])

    batched_tensor = torch.cat(flats, dim=0)
    batched_sif = sif.Sif.from_flat_tensor(batched_tensor, symcs[0])
    unbatched = batched_sif 
    





if __name__ == '__main__':
  unittest.main()

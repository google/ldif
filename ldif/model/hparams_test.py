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
"""Tests for ldif.hparams."""

import tempfile

from tensorflow.python.platform import googletest

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.model import hparams
# pylint: enable=g-bad-import-order


class HparamsTest(googletest.TestCase):

  def test_read_and_write(self):
    with tempfile.NamedTemporaryFile(delete=False) as fname:
      hparams.write_hparams(hparams.autoencoder_hparams, fname.name)
      hps = hparams.read_hparams(fname.name)
    self.assertEqual(repr(hparams.autoencoder_hparams), repr(hps))


if __name__ == '__main__':
  googletest.main()

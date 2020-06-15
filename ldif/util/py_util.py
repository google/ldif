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
"""General python utility functions."""

import contextlib
import functools
import os
import shutil
import subprocess as sp
import tempfile

import numpy as np


def compose(*fs):
  composition = lambda f, g: lambda x: f(g(x))
  identity = lambda x: x
  return functools.reduce(composition, fs, identity)


def maybe(x, f):
  """Returns [f(x)], unless f(x) raises an exception. In that case, []."""
  try:
    result = f(x)
    output = [result]
  # pylint:disable=broad-except
  except Exception:
    # pylint:enable=broad-except
    output = []
  return output


@contextlib.contextmanager
def py2_temporary_directory():
  d = tempfile.mkdtemp()
  try:
    yield d
  finally:
    shutil.rmtree(d)


@contextlib.contextmanager
def x11_server():
  """Generates a headless x11 target to use."""
  idx = np.random.randint(10, 10000)
  prev_display_name = os.environ['DISPLAY']
  x11 = sp.Popen('Xvfb :%i' % idx, shell=True)
  os.environ['DISPLAY'] = ':%i' % idx
  try:
    yield idx
  finally:
    x11.kill()
    os.environ['DISPLAY'] = prev_display_name


def merge(x, y):
  z = x.copy()
  z.update(y)
  return z


def merge_into(x, ys):
  out = []
  for y in ys:
    out.append(merge(x, y))
  return out

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
"""Utilities for managing gpus."""

import sys

import subprocess as sp

from ldif.util.file_util import log


def get_free_gpu_memory(cuda_device_index):
  """Returns the current # of free megabytes for the specified device."""
  if sys.platform == "darwin":
    # No GPUs on darwin...
    return 0
  result = sp.check_output('nvidia-smi --query-gpu=memory.free '
                           '--format=csv,nounits,noheader',
                           shell=True)
  result = result.decode('utf-8').split('\n')[:-1]
  log.verbose(f'The system has {len(result)} gpu(s).')
  free_mem = int(result[cuda_device_index])
  log.info(f'The {cuda_device_index}-th GPU has {free_mem} MB free.')
  if cuda_device_index >= len(result):
    raise ValueError(f"Couldn't parse result for GPU #{cuda_device_index}")
  return int(result[cuda_device_index])


def get_allowable_fraction_without(mem_to_reserve, cuda_device_index):
  """Returns the fraction to give to tensorflow after reserving x megabytes."""
  current_free = get_free_gpu_memory(cuda_device_index)
  allowable = current_free - mem_to_reserve  # 1GB
  allowable_fraction = allowable / current_free
  if allowable_fraction <= 0.0:
    raise ValueError(f"Can't leave 1GB over for the inference kernel, because"
                     f" there is only {allowable} total free GPU memory.")
  return allowable_fraction

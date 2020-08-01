#!/bin/bash
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
cd $(dirname $0)

# Get the cuda version to figure out which architectures
# to build for:
version=($(python get_cuda_version.py))
major_version=${version[0]}
minor_version=${version[1]}
echo "Major version is ${major_version} and minor version is ${minor_version}"

# Support for Pascal (10-series cards, P100). Requires at least CUDA 8.
# We don't support less than this.
targets="-gencode=arch=compute_60,code=sm_60 \
  -gencode=arch=compute_61,code=sm_61 \
  -gencode=arch=compute_62,code=sm_62"

# Support for Volta (e.g. V100). Requires CUDA 9.
if [[ $major_version -ge 9 ]]; then
  echo "Adding CUDA 9 Targets."
  targets="${targets} \
    -gencode=arch=compute_70,code=sm_70 \
    -gencode=arch=compute_72,code=sm_72"
fi

# Support for Turing (e.g. RTX 2080 Ti). Requires CUDA 10.
if [[ $major_version -ge 10 ]]; then
  echo "Adding CUDA 10 Targets."
  targets="${targets} \
    -gencode=arch=compute_75,code=sm_75"
fi

nvcc -Xptxas -O3 \
  ${targets} \
  --ptxas-options=-v -maxrregcount 63 $1 \
  ldif2mesh.cu -o ldif2mesh

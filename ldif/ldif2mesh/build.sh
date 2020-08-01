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
nvcc -Xptxas -O3 \
  --generate-code=arch=compute_60,code=sm_60 \
  --generate-code=arch=compute_61,code=sm_61 \
  --generate-code=arch=compute_62,code=sm_62 \
  --generate-code=arch=compute_70,code=sm_70 \
  --generate-code=arch=compute_72,code=sm_72 \
  --generate-code=arch=compute_75,code=sm_75 \
  --ptxas-options=-v -maxrregcount 63 $1 \
  ldif2mesh.cu -o ldif2mesh

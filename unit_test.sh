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

# This script trains on two open-source meshes, computes metrics, and then
# visualizes the reconstructed results.

set -e
set -v

# First, make the dataset. mktemp should be available by default
# on linux + macos
d=$(mktemp -d)
echo $d
# Do everything in the temp directory, just to be tidy.
mkdir -p ${d}/input_meshes/train/animal/
bd=$(dirname $0)

cp ${bd}/ldif/test_data/bob.ply ${bd}/ldif/test_data/blub.ply \
  ${d}/input_meshes/train/animal/

python meshes2dataset.py --mesh_directory ${d}/input_meshes \
  --dataset_directory ${d}/output_dataset

# Need a batch size of 2 due to batch-norm, hence two meshes.
# The step count is set to 1000, which isn't nearly enough
# to fully converge. But it is enough to at least tell
# if things are working while keeping the unit test relatively
# short. To get convergence, set to ~50k steps.
python train.py --batch_size 2 --experiment_name two-shape-ldif \
  --model_directory ${d}/models --model_type "ldif" \
  --dataset_directory ${d}/output_dataset --train_step_count 1000 \
  --log_level info

python eval.py --dataset_directory ${d}/output_dataset \
  --model_directory ${d}/models --experiment_name two-shape-ldif \
  --split train --log_level verbose --use_inference_kernel --result_directory \
  ${d}/results --save_ldifs --save_results --save_meshes --visualize



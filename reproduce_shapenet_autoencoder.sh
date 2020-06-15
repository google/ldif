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

# This script will reproduce the shapenet autoencoding experiment from the LDIF
# paper. It is not really meant to be run directly (it would probably be better
# to run the commands one at a time as applicable), but just to show how the
# code fits together. Note that each of the scripts called by this code has
# its own documentation, and the pipeline is also described in the README.

# This script expects that the environment has been set up according to the
# README, and that there is a directory of watertight meshes set up as follows:
# <ldif_root>/input_meshes/shapenet/{train,test,val}/<class>/*.ply.

# The following script only replicates the shapenet experiment if there are 13
# classes in each of train, test, and val, split according to the paper's split
# (provided in paper_splits.txt), and they have already been processed to be
# watertight via OccNet's code. There should be a total of 43757 meshes.

# This code trains on the stanford bunny computes metrics, and then visualizes
# the reconstructed mesh.

set -e
set -v

# Get the path to the LDIF code.
bd=$(dirname $0)
dataset=${bd}/shapenet
models=${bd}/trained_models
results=${bd}/shapenet-results
# First, make the dataset

python meshes2dataset.py --mesh_directory ${bd}/input_meshes \
  --dataset_directory $dataset

python train.py --batch_size 24 --experiment_name shapenet-ldif \
  --model_directory $models --model_type "ldif" \
  --dataset_directory $dataset

python eval.py --dataset_directory $dataset --model_directory $models \
  --experiment_name single-shape-ldif --split test --log_level verbose \
  --use_inference_kernel --result_directory $results --save_ldifs \
  --save_results --save_meshes --visualize




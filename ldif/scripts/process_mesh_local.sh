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

set -e

mesh_in=$1
outdir=$2
ldif_root=$3

dodeca_path=${ldif_root}/data/dodeca_cameras.cam
conf_path=${ldif_root}/data/base_conf.conf
gaps=${ldif_root}/gaps/bin/x86_64/

# On macos osmesa is not used, on linux it is:
if [[ $(uname -s) == Darwin* ]]
then
  mesa=""
else
  mesa="-mesa"
fi

mkdir -p $outdir || true

mesh_orig=${outdir}/mesh_orig.${mesh_in##*.}
ln -s $mesh_in $mesh_orig

mesh=${outdir}/model_normalized.obj
# Step 0) Normalize the mesh before applying all other operations.
${gaps}/msh2msh $mesh_orig $mesh -scale_by_pca -translate_by_centroid \
  -scale 0\.25 -debug_matrix ${outdir}/orig_to_gaps.txt

# Step 1) Generate the coarse inside/outside grid:
${gaps}/msh2df $mesh ${outdir}/coarse_grid.grd -bbox -0\.7 -0\.7 -0\.7 0\.7 \
  0\.7 0\.7 -border 0 -spacing 0\.044 -estimate_sign -v

# Step 2) Generate the near surface points:
${gaps}/msh2pts $mesh ${outdir}/nss_points.sdf -near_surface -max_distance \
  0\.04 -num_points 100000 -v -binary_sdf # -curvature_exponent 0

# Step 3) Generate the uniform points:
${gaps}/msh2pts $mesh ${outdir}/uniform_points.sdf -uniform_in_bbox -bbox \
  -0\.7 -0\.7 -0\.7 0\.7 0\.7 0\.7 -npoints 100000 -binary_sdf

# Step 4) Generate the depth renders:
depth_dir=${outdir}/depth_images/
${gaps}/scn2img $mesh $dodeca_path $depth_dir -capture_depth_images \
  $mesa -width 224 -height 224

# The normalized mesh is no longer needed on disk; we have the transformation,
# so if we need it we can load the original symlinked mesh and transform it
# to the normalized frame.
rm $mesh

local_conf=${outdir}/custom_conf.conf
echo "dataset_processed" > $local_conf
echo "depth_directory ${depth_dir}" >> $local_conf
cat $conf_path >> $local_conf
# TODO(kgenova) We have to write out normals as well?
${gaps}/conf2img $local_conf ${outdir}/normals \
  -create_normal_images -width 224 -height 224 $mesa
rm $local_conf

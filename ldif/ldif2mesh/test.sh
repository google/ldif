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
set -x

gaps=../gaps/bin/x86_64/
rm test-ldif-output.ply || true
rm test-sif-output.ply || true
./build.sh
./ldif2mesh test-ldif.txt extracted.occnet test-grid.grd -resolution 128
${gaps}/grd2msh test-grid.grd test-ldif-output.ply -threshold -0.07
rm test-grid.grd
./ldif2mesh test-sif.txt extracted.occnet test-grid.grd -resolution 128
${gaps}/grd2msh test-grid.grd test-sif-output.ply -threshold -0.07
rm test-grid.grd
${gaps}/mshview test-ldif-output.ply test-sif-output.ply -back

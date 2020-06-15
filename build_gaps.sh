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
set -v

# We're going to need these dependencies, start with that.
sudo apt install mesa-common-dev libglu1-mesa-dev libosmesa6-dev

# If the above command fails, get the GL/gl.h and GL/glu.h headers, delete the
# above line, and try again.

# CD inside the ldif/ldif python package
cd ldif

# This should create a gaps/ folder at ldif/ldif/gaps/
git clone https://github.com/tomfunkhouser/gaps.git

# This should make a copy of the qview folder at ldif/ldif/gaps/apps/qview/
cp -R ./qview gaps/apps/

# Make GAPS (assuming 8 threads):
cd gaps
make mesa -j8

# ptsview isn't made by default, build it:
cd apps/ptsview
make mesa -j8

# The GAPS scripts don't know about qview, make it too:
cd ../qview
make mesa -j8


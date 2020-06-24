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

# CD inside the ldif/ldif python package
cd ldif

# This should create a gaps/ folder at ldif/ldif/gaps/
if [[ -d gaps ]]
then
  echo "GAPS has already been cloned to ldif/ldif/gaps, skipping."
else
  git clone https://github.com/tomfunkhouser/gaps.git
fi

# Necessary dependencies:
# Figure out whether we are on MacOS or Linux:
if [[ $(uname -s) == Darwin* ]]
then
  echo "On MacOS, GL dependencies should have shipped and OSMesa support is disabled."
else
  # On linux, the packages need to be installed.
  sudo apt-get install mesa-common-dev libglu1-mesa-dev libosmesa6-dev libxi-dev libgl1-mesa-dev
  # For some reason on Ubuntu there can be a broken link from /usr/lib/x86_64-linux-gnu/libGL.so
  # to libGL.so.1.2.0 in the same directory, which does not exist. However libgl1-mesa-glx should
  # provide libGL.so.1.2.0. Reinstalling libgl1-mesa-glx results in libGL.so.1.2.0 correctly
  # existing in /usr/lib/x86_64-linux-gnu as it should.
  sudo apt-get install --reinstall libgl1-mesa-glx
fi
# If the above command(s) fail, get the GL/gl.h and GL/glu.h headers, OSMesa and GL
# static libraries (osmesa on macos), delete the above code, and try again.

# Now apply customizations to GAPS:

# This should make a copy of the qview folder at ldif/ldif/gaps/apps/qview/
if [[ -d gaps/apps/qview ]]
then
  echo "qview has already been copied into ldif/ldif/gaps/qview, skipping."
else
  cp -R ./qview gaps/apps/
fi

# Everything is local to GAPS from this point:
cd gaps

# Ptsview and qview aren't built by default, adjust the makefile to build them.
# sed commands are for script idempotency
sed -i.bak '/ptsview/d' ./apps/Makefile
sed -i.bak '/qview/d' ./apps/Makefile
echo "	cd ptsview; \$(MAKE) \$(TARGET)" >> ./apps/Makefile
echo "	cd qview; \$(MAKE) \$(TARGET)" >> ./apps/Makefile

# Make GAPS (assuming 8 threads):
# On MacOS, using OSMesa is more difficult, so we don't
if [[ ! $(uname -s) == Darwin* ]]
then
  make mesa -j8
else
  make -j8
fi

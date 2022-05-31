#!/bin/bash
# Copyright 2022 Google LLC
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


# Don't run this directly if your shell isn't bash or if conda isn't
# configured to work in bash scripts. Instead, just run the conda and pip commands one at a time in your interactive shell.

set -e
set -x

conda create --name ldif-depth python=3.6
conda activate ldif-depth
python -m pip install -r depth_requirements.txt


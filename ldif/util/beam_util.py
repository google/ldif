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
"""Utilities for apache beam jobs."""

import apache_beam as beam


def filter_success(elt):
  return [] if 'FAILED' in elt else [elt]


def filter_failure(elt):
  return [elt] if 'FAILED' in elt else []


def report_errors(elts):
  errs = []
  for elt in elts:
    errs.append(f"{elt['mesh_identifier']}|||{elt['FAILED']}\n")
  return '\n'.join(errs) + '\n'


def map_and_report_failures(inputs, f, name, fail_base, applier=None):
  """Applies a function and then parses out the successes and failures."""
  if applier is None:
    applier = beam.Map
  mapped = inputs | name >> applier(f)
  failed_to_map = mapped | f'Get{name}Failed' >> beam.FlatMap(filter_failure)
  successfully_mapped = mapped | f'Get{name}Succeeded' >> beam.FlatMap(
      filter_success)
  fail_file = fail_base + f'_failed_to_{name}.txt'
  _ = (
      failed_to_map | f'CombineErrPcollFor{name}' >> beam.combiners.ToList()
      | f'MakeErrStringFor{name}' >> beam.Map(report_errors)
      | f'WritErrStringFor{name}' >> beam.io.WriteToText(
          fail_file, num_shards=1, shard_name_template=''))
  return successfully_mapped

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
"""Extracts results for paper presentation."""

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import pandas as pd

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.inference import metrics
from ldif.results import results_pb2
from ldif.util import file_util
from ldif.util import mesh_util
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order

FLAGS = flags.FLAGS

flags.DEFINE_string('occnet_dir', None, "The folder with OccNet's meshes.")
flags.mark_flag_as_required('occnet_dir')

flags.DEFINE_boolean('write_metrics', None,
                     'Whether to write full metrics in the results.')
flags.mark_flag_as_required('write_metrics')

flags.DEFINE_boolean('write_metric_summaries', None,
                     'Whether to write summaries of metrics in the results.')
flags.mark_flag_as_required('write_metric_summaries')

flags.DEFINE_boolean('transform', True,
                     'Whether to transform from the gaps to occnet frame.')


def make_metrics(proto):
  """Builds a dictionary containing proto elements."""
  key, s = proto
  p = results_pb2.Results.FromString(s)
  mesh_path = FLAGS.occnet_dir + key.replace('test/', '') + '.ply'
  log.warning('Mesh path: %s' % mesh_path)
  try:
    mesh = file_util.read_mesh(mesh_path)
    if FLAGS.transform:
      # TODO(ldif-user) Set up the path to the transformation:
      tx_path = 'ROOT_DIR/%s/occnet_to_gaps.txt' % key
      occnet_to_gaps = file_util.read_txt_to_np(tx_path).reshape([4, 4])
      gaps_to_occnet = np.linalg.inv(occnet_to_gaps)
      mesh.apply_transform(gaps_to_occnet)
  # pylint: disable=broad-except
  except Exception as e:
    # pylint: enable=broad-except
    log.error("Couldn't load %s, skipping due to %s." % (mesh_path, repr(e)))
    return []

  gt_mesh = mesh_util.deserialize(p.gt_mesh)
  dir_out = FLAGS.occnet_dir + '/metrics-out-gt/%s' % key
  if not file_util.exists(dir_out):
    file_util.makedirs(dir_out)
  file_util.write_mesh(f'{dir_out}gt_mesh.ply', gt_mesh)
  file_util.write_mesh(f'{dir_out}occnet_pred.ply', mesh)

  nc, fst, fs2t, chamfer = metrics.all_mesh_metrics(mesh, gt_mesh)
  return [{
      'key': key,
      'Normal Consistency': nc,
      'F-Score (tau)': fst,
      'F-Score (2*tau)': fs2t,
      'Chamfer': chamfer,
  }]


def save_metrics(elts):
  elts = pd.DataFrame(elts)
  csv_str = elts.to_csv()
  return csv_str


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  root_out = FLAGS.occnet_dir + '/extracted'
  if not file_util.exists(root_out):
    file_util.mkdir(root_out)

  if FLAGS.write_metrics:
    # TODO(ldif-user): Set up your own pipeline runner
    # TODO(ldif-user) Replace lambda x: None with a proto reader.
    with beam.Pipeline() as p:
      protos = p | 'ReadResults' >> (lambda x: None)

      with_metrics = protos | 'ExtractMetrics' >> beam.FlatMap(make_metrics)
      result_pcoll = with_metrics | 'MakeMetricList' >> (
          beam.combiners.ToList())
      result_str = result_pcoll | 'MakeMetricStr' >> beam.Map(save_metrics)
      out_path = FLAGS.occnet_dir + '/extracted/metrics_ub-v2.csv'
      _ = result_str | 'WriteMetrics' >> beam.io.WriteToText(
          out_path, num_shards=1, shard_name_template='')
  if FLAGS.write_metric_summaries:
    log.info('Aggregating results locally.')
    result_path = FLAGS.occnet_dir + '/extracted/metrics_ub-v2.csv'
    final_results = metrics.aggregate_extracted(result_path)
    summary_out_path = result_path.replace('/metrics_ub-v2.csv',
                                           '/metric_summary_ub-v2.csv')
    file_util.writetxt(summary_out_path, final_results.to_csv())


if __name__ == '__main__':
  app.run(main)

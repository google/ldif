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

import functools

from absl import app
from absl import flags
import apache_beam as beam
import pandas as pd
import tqdm

# pylint: disable=g-bad-import-order
from ldif.inference import metrics
from ldif.inference import util as inference_util
from ldif.results import results_pb2
from ldif.util import file_util
from ldif.util import mesh_util
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order

FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', None, 'The input directory.')
flags.mark_flag_as_required('input_dir')

flags.DEFINE_string('xids', None, 'The XIDs to evaluate on.')
flags.mark_flag_as_required('xids')

flags.DEFINE_boolean('use_newest', None,
                     'Whether to use the newest checkpoint per XID.')
flags.mark_flag_as_required('use_newest')

flags.DEFINE_boolean('write_results', None,
                     'Whether to write a directory of the results.')
flags.mark_flag_as_required('write_results')

flags.DEFINE_boolean('write_metrics', None,
                     'Whether to write full metrics in the results.')
flags.mark_flag_as_required('write_metrics')

flags.DEFINE_boolean('write_metric_summaries', None,
                     'Whether to write summaries of metrics in the results.')
flags.mark_flag_as_required('write_metric_summaries')


def _write_results(proto, xid=None):
  """Writes the prediction, ground truth, and representation to disk."""
  key, s = proto
  p = results_pb2.Results.FromString(s)
  if xid is None:
    dir_out = FLAGS.input_dir + '/extracted/' + key + '/'
  else:
    dir_out = FLAGS.input_dir + '/extracted/XID%i/%s/' % (xid, key)
  file_util.makedirs(dir_out)
  file_util.write_mesh(f'{dir_out}/gt_mesh.ply', p.gt_mesh)
  file_util.write_mesh(f'{dir_out}/pred_mesh.ply', p.mesh)
  file_util.writetxt(f'{dir_out}/sif.txt', p.representation)
  # TODO(ldif-user) Set up the unnormalized2normalized path.
  path_to_tx = '/ROOT_DIR/%s/occnet_to_gaps.txt' % key
  occnet_to_gaps = file_util.read_txt_to_np(path_to_tx).reshape([4, 4])
  pm = mesh_util.deserialize(p.mesh)
  pm.apply_transform(occnet_to_gaps)
  file_util.write_mesh(f'{dir_out}/nrm_pred_mesh.ply', pm)
  gtm = mesh_util.deserialize(p.gt_mesh)
  gtm.apply_transform(occnet_to_gaps)
  file_util.write_mesh(f'{dir_out}/nrm_gt_mesh.ply', gtm)


def write_results(proto, xid=None):
  try:
    return [_write_results(proto, xid)]
  # pylint: disable=broad-except
  except Exception:
    # pylint: enable=broad-except
    return []


def make_metrics(proto):
  key, s = proto
  p = results_pb2.Results.FromString(s)
  mesh = mesh_util.deserialize(p.mesh)
  gt_mesh = mesh_util.deserialize(p.gt_mesh)
  nc, fst, fs2t, chamfer = metrics.all_mesh_metrics(mesh, gt_mesh)
  return {
      'key': key,
      'Normal Consistency': nc,
      'F-Score (tau)': fst,
      'F-Score (2*tau)': fs2t,
      'Chamfer': chamfer,
      'IoU': p.iou
  }


def save_metrics(elts):
  elts = pd.DataFrame(elts)
  csv_str = elts.to_csv()
  return csv_str


def get_result_path(xid):
  """Generates the result path associated with the requested XID."""
  # TODO(ldif-user) Set up the result path:
  base = FLAGS.input_dir + '/ROOT%i-*00000-*' % xid
  matches = file_util.glob(base)
  assert len(matches) >= 1
  ckpts = []
  for match in matches:
    # TODO(ldif-user) Set the file extension
    extension = None
    ckpt = int(match.split(extension)[0].split('-')[-1])
    ckpts.append(ckpt)
  if len(ckpts) > 1 and not FLAGS.use_newest:
    log.info('Found multiple checkpoint matches for %s and --nouse_newest: %s' %
             (base, repr(ckpts)))
  if len(ckpts) == 1:
    ckpt = ckpts[0]
  elif len(ckpts) > 1:
    ckpts.sort()
    ckpt = ckpts[-1]
    log.info('Found multiple checkpoint matches %s, using %s' %
             (repr(ckpts), repr(ckpt)))
  # TODO(ldif-user) Set up the result path:
  path = FLAGS.input_dir + '/ROOT%i-%i.*'
  path = path % (xid, ckpt)
  return path


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  xids = inference_util.parse_xid_str(FLAGS.xids)

  root_out = FLAGS.input_dir + '/extracted'
  if not file_util.exists(root_out):
    file_util.mkdir(root_out)

  if FLAGS.write_metrics or FLAGS.write_results:
    # TODO(ldif-user): Set up your own pipeline runner
    with beam.Pipeline() as p:
      for xid in xids:
        name = 'XID%i' % xid
        path = get_result_path(xid)
        # TODO(ldif-user) Replace lambda x: None with a proto reader.
        protos = p | 'ReadResults%s' % name >> (lambda x: None)

        if FLAGS.write_results:
          map_fun = functools.partial(write_results, xid=xid)
          _ = protos | 'ExtractResults%s' % name >> beam.FlatMap(map_fun)
        if FLAGS.write_metrics:
          with_metrics = protos | 'ExtractMetrics%s' % name >> beam.Map(
              make_metrics)
          result_pcoll = with_metrics | 'MakeMetricList%s' % name >> (
              beam.combiners.ToList())
          result_str = result_pcoll | 'MakeMetricStr%s' % name >> beam.Map(
              save_metrics)
          out_path = FLAGS.input_dir + '/extracted/%s_metrics-v2.csv' % name
          _ = result_str | 'WriteMetrics%s' % name >> beam.io.WriteToText(
              out_path, num_shards=1, shard_name_template='')
  if FLAGS.write_metric_summaries:
    log.info('Aggregating results locally.')
    for xid in tqdm.tqdm(xids):
      result_path = FLAGS.input_dir + '/extracted/XID%i_metrics-v2.csv' % xid
      final_results = metrics.aggregate_extracted(result_path)
      summary_out_path = result_path.replace('_metrics-v2.csv',
                                             '_metric_summary-v2.csv')
      file_util.writetxt(summary_out_path, final_results.csv())

if __name__ == '__main__':
  app.run(main)

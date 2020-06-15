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
"""Prints a merged summary table from a variety of XIDs."""

from absl import app
from absl import flags

import pandas as pd
import tabulate

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.inference import util as inference_util
from ldif.util import file_util
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order


FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', None, 'The input directory with results.')
flags.mark_flag_as_required('input_dir')

flags.DEFINE_string('xids', None, 'The XIDs to evaluate on.')
flags.mark_flag_as_required('xids')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  xids = inference_util.parse_xid_str(FLAGS.xids)
  log.info(f'XIDS: {xids}')

  names = []
  ious = []
  fscores = []
  chamfers = []
  xid_to_name = {
      1: 'CRS SIF',
      2: 'CRS ldif',
      3: 'CR SIF',
      4: 'CR ldif',
      5: 'CRS PT SIF',
      6: 'CRS PT ldif',
      7: 'CR PT SIF',
      8: 'CR PT ldif'
  }
  for xid in xids:
    path = f'{FLAGS.input_dir}/extracted/XID{xid}_metric_summary-v2.csv'
    df = file_util.read_csv(path)
    log.info(f'XID {xid}:')
    log.info(df)
    mean = df[df['class'].str.contains('mean')]
    names.append(xid_to_name[xid])
    ious.append(float(mean['IoU']))
    fscores.append(float(mean['F-Score (tau)']))
    chamfers.append(float(mean['Chamfer']))
  l = list(zip(names, ious, fscores, chamfers))
  log.info('Start')
  log.info(names)
  log.info(ious)
  log.info(fscores)
  log.info(chamfers)
  log.info('End')
  df = pd.DataFrame(l, columns=['Name', 'IoU', 'F-Score (tau)', 'Chamfer'])
  log.info(df)
  pp = tabulate.tabulate(
      df, headers='keys', tablefmt='fancy_grid', floatfmt='.3f')
  log.info(pp)


if __name__ == '__main__':
  app.run(main)

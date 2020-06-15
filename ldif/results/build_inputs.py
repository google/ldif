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
"""Builds the worker tasks for the beam job."""

import sys

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.inference import experiment
from ldif.inference import util as inference_util
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order


def generate_models(flags):
  """Makes all the dictionaries that workers can process in parallel."""

  # TODO(kgenova) If the code fails here, add a model_dir argument.
  cur_experiment = experiment.Experiment(flags.model_dir,
                                         flags.model_name,
                                         flags.experiment_name)

  xids = inference_util.parse_xid_str(flags.xids)
  cur_experiment.filter_jobs_by_xid(xids)
  cur_experiment.set_order(xids)
  log.info('XIDs to run on: %s' % repr(cur_experiment.visible_xids))

  if not (flags.ckpt or flags.xid_to_ckpt):
    log.info('Choose a checkpoint and rerun:')
    for job in cur_experiment.visible_jobs:
      if not job.has_at_least_one_checkpoint:
        log.info('XID %i: No checkpoints written.' % job.xid)
      else:
        log.info('XID %i: %s' % (job.xid, repr(job.all_checkpoint_indices)))
    sys.exit(1)

  for job in cur_experiment.visible_jobs:
    if not job.has_at_least_one_checkpoint:
      log.info('Checkpoint(s) were specified but one of the XIDs, %i, has no'
               ' checkpoints visible. Please remove it and rerun.' % job.xid)
      sys.exit(1)

  if flags.xid_to_ckpt:
    xid_to_ckpt = inference_util.parse_xid_to_ckpt(flags.xid_to_ckpt)
    for job in cur_experiment.visible_jobs:
      if job.xid not in xid_to_ckpt:
        log.info(('XID -> Checkpoint map was specified but visible XID %i has'
                  ' no checkpoint.') % job.xid)
        sys.exit(1)

    for xid in list(xid_to_ckpt.keys()):
      job = cur_experiment.job_with_xid(xid)
      available_checkpoints = job.all_checkpoint_indices
      ckpt = xid_to_ckpt[xid]
      if ckpt not in available_checkpoints:
        log.info(('XID -> Checkpoint map was specified but mapping %i -> %i is '
                  'invalid because checkpoint %i does not exist.') %
                 (xid, ckpt, ckpt))
        sys.exit(1)
  else:
    if not flags.ckpt:
      raise RuntimeError('Internal error: passed checkpoint check but'
                         'no checkpoint passed.')
    xid_to_ckpt = {}
    for job in cur_experiment.visible_jobs:
      xid_to_ckpt[job.xid] = job.latest_checkpoint_before(
          flags.ckpt, must_equal=False).idx

  models = []
  for job in cur_experiment.visible_jobs:
    ckpt = experiment.Checkpoint(job, xid_to_ckpt[job.xid])
    models.append({
        'xid': job.xid,
        'ckpt': ckpt.idx,
        'user': cur_experiment.user,
        'model_name': cur_experiment.model_name,
        'experiment_name': cur_experiment.experiment_name,
    })
  return models


def generate_tasks(flags):
  """Generates the set of evaluations that need to be done."""
  if 'shapenet' in flags.dataset:
    identifiers = inference_util.get_mesh_identifiers(flags.split,
                                                      flags.category)
    log.info(identifiers[0])
    log.info('There are %i identifiers' % len(identifiers))
    log.info('The instance count is %i' % flags.instance_count)
    if flags.instance_count != -1:
      assert flags.instance_count <= len(identifiers)
      identifiers = identifiers[:flags.instance_count]
  out = []
  for identifier in identifiers:
    out.append({
        'mesh_identifier': identifier,
        'dataset': flags.dataset,
        'split': flags.split,
    })
  return out

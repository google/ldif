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
"""A wrapper for getting results from an experiment."""

import importlib
import os
import numpy as np

# ldif is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.model import hparams as hparams_util
from ldif.util import file_util
from ldif.util import path_util
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order

importlib.reload(hparams_util)


def to_xid(job):
  assert '-' in job
  return int(job.split('-')[0])


class Checkpoint(object):
  """A single checkpoint of a job."""

  def __init__(self, parent_job, idx):
    self.idx = idx
    self.job = parent_job

  @property
  def relpath(self):
    return 'model.ckpt-%i' % self.idx

  # def get_ckpt_dir(self, job):
  @property
  def directory(self):
    return '%s/train' % self.job.root_dir

  @property
  def abspath(self):
    return '%s/%s' % (self.directory, self.relpath)


class Job(object):
  """A single hparam combo within an experiment."""

  def __init__(self, parent_experiment, job_str, use_temp_ckpts='warn'):
    self.experiment = parent_experiment
    self.job_str = job_str
    self.xid = to_xid(job_str)
    self._use_temp_ckpts = use_temp_ckpts
    self.lazy_attrs = [
        '_all_checkpoints', '_root_dir', '_hparams', '_model_config'
    ]
    for a in self.lazy_attrs:
      setattr(self, a, None)

  @classmethod
  def from_xid(cls, parent_experiment, xid):
    valid_job_strs = [x for x in parent_experiment.job_strs if to_xid(x) == xid]
    assert len(valid_job_strs) == 1
    job_str = valid_job_strs[0]
    return cls(parent_experiment, job_str)

  def _ensure_lazy_attrs_are_zero(self):
    for a in self.lazy_attrs:
      val = getattr(self, a)
      is_initialized = val is not None
      if is_initialized:
        raise ValueError('Attribute %s has already been set to value %s' %
                         (a, repr(val)))

  def set_use_temp_ckpts(self, useable):
    assert useable in [True, False, 'warn']
    self._ensure_lazy_attrs_are_zero()
    self._use_temp_ckpts = useable

  @property
  def is_visible(self):
    return self.xid in self.experiment.visible_xids

  def ensure_visible(self):
    if not self.is_visible:
      raise ValueError(
          'Job failed ensure_visible() check: %i. Only %i are visible.' %
          (self.xid, self.experiment.visible_xids))

  @property
  def ckpt_dir(self):
    return self.root_dir + '/train/'

  @property
  def root_dir(self):
    if self._root_dir is None:
      self._root_dir = '%s/%s/' % (self.experiment.root, self.job_str)
      if not file_util.exists(self._root_dir):
        raise ValueError("Couldn't find job directory at %s." % self._root_dir)
    return self._root_dir

  @property
  def all_checkpoint_indices(self):
    return [c.idx for c in self.all_checkpoints]

  @property
  def all_checkpoints(self):
    """A list of all checkpoint objects in the checkpoint directory."""
    if self._all_checkpoints is None:
      candidates = [
          os.path.basename(n) for n in file_util.glob(f'{self.ckpt_dir}/*')
      ]
      inds = [
          int(x.replace('model.ckpt-', '').replace('.index', ''))
          for x in candidates
          if 'index' in x and 'tempstate' not in x
      ]
      inds.sort(reverse=False)
      # The train task may delete the 5 most recent checkpoints periodically:
      if not inds:
        raise ValueError('There are no checkpoints in the directory %s.' %
                         self.ckpt_dir)
      elif self._use_temp_ckpts is True:  # pylint: disable=g-bool-id-comparison
        message = 'Temporary checkpoints are enabled.'
        message += ' The most recent temporary checkpoint is %i.' % inds[-1]
        if len(inds) >= 6:
          message += ' The most recent permanent checkpoint is %i.' % inds[-6]
        else:
          message += ' There are no permanent checkpoints.'
        log.warning(message)
      elif len(inds) < 6:
        if self._use_temp_ckpts == 'warn':
          warning = (
              'No permanent checkpoints. Resorting to temporary one: %i' %
              inds[-1])
          log.warning(warning)
          inds = [inds[-1]]
        elif not self._use_temp_ckpts:
          raise ValueError(
              'Only temporary checkpoints are available, and they are not enabled.'
          )
      else:
        inds = inds[:-5]
      self._all_checkpoints = [Checkpoint(self, ind) for ind in inds]
    return self._all_checkpoints

  def latest_checkpoint_before(self, idx, must_equal=False):
    """Returns the checkpoint object for the most recent checkpoint."""
    if idx == -1:
      c = self.newest_checkpoint
    else:
      cands = [c for c in self.all_checkpoints if c.idx <= idx]
      if not cands:
        raise ValueError('No checkpoint present before %i. All indices: %s' %
                         (idx, repr(self.all_checkpoint_indices)))
      c = cands[-1]  # Pre-sorted.
    if must_equal and c.idx != idx:
      raise ValueError(
          'Couldn\'t find exactly requested checkpoint %i: %i chosen.' %
          (idx, c.idx))
    return c

  @property
  def has_at_least_one_checkpoint(self):
    return bool(self.all_checkpoints)

  def ensure_has_at_least_one_checkpoint(self):
    if not self.has_at_least_one_checkpoint:
      raise ValueError('There are no checkpoints for this job (xid %i).' %
                       self.xid)

  @property
  def checkpoint_count(self):
    return len(self.all_checkpoints)

  @property
  def newest_checkpoint(self):
    self.ensure_has_at_least_one_checkpoint()
    return self.all_checkpoints[-1]  # Sorted

  def k_evenly_spaced_checkpoints(self, k):
    """Returns k checkpoint objects distributed approximately evenly in step."""
    if k == 1:
      return self.newest_checkpoint
    elif k <= 0:
      raise ValueError('Invalid k: %i. k must be >= 1.' % k)
    elif self.checkpoint_count < 2 * k:
      raise ValueError(
          "Can't selected %i checkpoints when %i < 2*%i are present." %
          (k, self.checkpoint_count, k))
    else:
      chosen = np.linspace(start=0, stop=self.checkpoint_count, num=k)
      chosen = np.clip(chosen.astype(np.int32), 0, self.checkpoint_count - 1)
      chosen = list(chosen)
      return [self.all_checkpoints[i] for i in chosen]

  @property
  def hparams(self):
    """Load a tf.HParams() object based on the serialized hparams file."""
    if self._hparams is None:
      hparam_path = '%s/train/hparam_pickle.txt' % self.root_dir
      if file_util.exists(hparam_path):
        log.info('Found serialized hparams. Loading from %s' % hparam_path)
        # hparams = hparams_util.read_hparams(hparam_path)
        self._hparams = (
            hparams_util.read_hparams_with_new_backwards_compatible_additions(
                hparam_path))
      else:
        raise ValueError('No serialized hparam file found at %s' % hparam_path)
    return self._hparams

  @property
  def model_config(self):
    if self._model_config is None:
      self._model_config = ModelConfig(self.hparams)
    return self._model_config


class Experiment(object):
  """A tensorflow experiment run. Provides an interface to results."""

  def __init__(self, model_dir, model_name, experiment_name):
    self.root = f'{model_dir}/{model_name}-{experiment_name}'
    self.model_name = model_name
    self.experiment_name = experiment_name

    if not file_util.exists(self.root):
      log.verbose('Regex expanding root to find experiment ID')
      options = file_util.glob(self.root[:-1] + '*')
      if len(options) != 1:
        log.verbose(
            "Tried to glob for directory but didn't find one path. Found:")
        log.verbose(options)
        raise ValueError('Directory not found: %s' % self.root)
      else:
        self.root = options[0] + '/'
        self.experiment_name = os.path.basename(self.root.strip('/'))
        self.experiment_name = self.experiment_name.replace(
            self.model_name + '-', '')
        log.verbose('Expanded experiment name with regex to root: %s' %
                    self.root)

    job_strs = [os.path.basename(n) for n in file_util.glob(f'{self.root}/*')]

    banned = ['log', 'mldash_config.txt', 'snapshot', 'mldash_config']
    job_strs = [p for p in job_strs if p not in banned]
    job_strs = sorted(job_strs, key=to_xid)
    log.verbose('Job strings: %s' % repr(job_strs))
    self.all_jobs = [Job(self, job_str) for job_str in job_strs]
    self._visible_jobs = self.all_jobs[:]

  def set_order(self, ordered_xids):
    """Reorders the visible jobs list."""
    # First, check that the input xids are a permutation of the visible ones.
    for xid in ordered_xids:
      if xid not in self.visible_xids:
        raise ValueError(
            'Trying to order xids with an unknown input xid: %i not in %i' %
            (xid, self.visible_xids))
    for xid in self.visible_xids:
      if xid not in ordered_xids:
        raise ValueError(
            'Trying to order xids without specifing xid: %i not in %i' %
            (xid, ordered_xids))
    xid_to_pos = {ordered_xids[i]: i for i in range(len(ordered_xids))}
    self._visible_jobs.sort(key=lambda j: xid_to_pos[j.xid])

  def job_with_xid(self, xid):
    jobs = [x for x in self.all_jobs if x.xid == xid]
    if len(jobs) != 1:
      raise ValueError('Expected one match for xid %i, but got: %s' %
                       (xid, repr(jobs)))
    return jobs[0]

  def filter_jobs_by_xid(self, xids):
    self._visible_jobs = [j for j in self._visible_jobs if j.xid in xids]
    return self.visible_jobs

  @property
  def visible_jobs(self):
    if self._visible_jobs:
      return self._visible_jobs
    else:
      raise ValueError('There are no remaining visible jobs.')

  @property
  def visible_job_count(self):
    return len(self.visible_jobs)

  @property
  def visible_xids(self):
    return [j.xid for j in self.visible_jobs]

  def job_from_xmanager_id(self, xid, must_be_visible=True):
    s = [j for j in self.all_jobs if j.xid == int(xid)]
    if len(s) != 1:
      raise ValueError(
          'Trouble matching xmanager id. Expected one match but got %s' %
          repr(s))
    j = s[0]
    if must_be_visible:
      j.ensure_visible()
    return j

  def job_from_job_str(self, job_str, must_be_visible=True):
    xid = to_xid(job_str)
    return self.job_from_xmanager_id(xid, must_be_visible)


class ResultStore(object):
  """A backing store for results.

  Has both remote and local components.

  This class is not synchronized. It assumes nothing is changing on the server
  or local disk except the operations it performs.
  """

  def __init__(self, experiment, desired_ckpts=-1):
    self.experiment = experiment
    self.available_ckpt_dict = {}
    self.desired_ckpts = desired_ckpts
    self.remote_base_dirs = {'train': {}, 'val': {}, 'test': {}}
    self._remote_mesh_names = None
    self._local_qual_path = None

  @property
  def local_root(self):
    return os.path.join(path_util.get_path_to_ldif_root(), 'result_store')

  @property
  def remote_root(self):
    raise NotImplementedError('Please pick a remote backing store.')

  def remote_result_ckpt_dir(self, xid):
    s = '%s/%s/%s/' % (self.remote_root, self.experiment.experiment_name,
                       self.experiment.job_from_xmanager_id(xid))
    return s

  def local_result_ckpt_dir(self, xid):
    s = '%s/%s/%i/' % (self.local_root, self.experiment.experiment_name, xid)
    return s

  def ckpts_for_xid_and_split(self, xid, split):
    """Returns the checkpoints with results for a given XID-split pair."""
    key = '%s-%s' % (str(xid), split)
    if key in self.available_ckpt_dict:
      return self.available_ckpt_dict[key]
    base = self.remote_result_ckpt_dir(xid)
    candidates = file_util.glob(f'{base}/*/{split}')
    ckpts = [int(x.split('/')[-2]) for x in candidates]
    self.available_ckpt_dict[key] = ckpts
    return ckpts

  def ckpt_for_xid(self, xid, split):
    """Gets the most desired checkpoint with results for the XID-split pair."""
    candidates = self.ckpts_for_xid_and_split(xid, split)
    assert candidates
    if self.desired_ckpts == -1:
      return max(candidates)
    else:
      if xid not in self.desired_ckpts:
        raise ValueError('Provided desired ckpts %s but no entry for %i' %
                         (repr(self.desired_ckpts), xid))
      desired = self.desired_ckpts[xid]
      if desired not in candidates:
        raise ValueError(
            'Provided desired ckpt %i but it does not exist on the remote store. Only %s exist.'
            % (desired, repr(candidates)))
      return desired

  def remote_result_base(self, xid, split):
    """Returns the base path for remote results for the xid-split pair."""
    if xid in self.remote_base_dirs[split]:
      return self.remote_base_dirs[split][xid]
    ckpt = self.ckpt_for_xid(xid, split)
    s = '%s/%i/%s' % (self.remote_result_ckpt_dir(xid), ckpt, split)
    # Ensure it's a valid directory:
    if not file_util.exists(s):
      raise ValueError(('No directory for split %s and ckpt %i for for xid %i.'
                        ' Expected path was: %s') % (split, ckpt, xid, s))
    self.remote_base_dirs[split][xid] = s
    return s

  def all_remote_mesh_names(self, split, ensure_nonempty=True):
    """Returns remote mesh hashes that are present for all xids."""
    if self._remote_mesh_names is not None:
      return self._remote_mesh_names
    all_mesh_names = None
    for xid in self.experiment.visible_xids:
      base = self.remote_result_base(xid, split)
      mesh_paths = file_util.glob(f'{base}/*/*.ply')
      if not mesh_paths and ensure_nonempty:
        raise ValueError('No meshes present for xid %i with path %s' %
                         (xid, base))
        # TODO(kgenova) Now we are assuming hashes are not replicated in
        # multiple synsets.
      mesh_names = set()
      for mesh_path in mesh_paths:
        mesh_hash = os.path.basename(mesh_path).replace('.ply', '')
        synset = mesh_path.split('/')[-2]
        mesh_names.add('%s-%s' % (synset, mesh_hash))
      if all_mesh_names is None:
        all_mesh_names = mesh_names
      else:
        all_mesh_names = all_mesh_names.intersection(mesh_names)
    if ensure_nonempty and not all_mesh_names:
      raise ValueError('There are 0 meshes common to the xids %i for split %s' %
                       (repr(self.experiment.visible_xids), split))
    self._remote_mesh_names = list(all_mesh_names)
    return list(all_mesh_names)

  def copy_meshes(self, mesh_names, split, overwrite_if_present=True):
    """Copies meshes from the remote store to the local cache."""
    for xid in self.experiment.visible_xids:
      log.verbose('Copying filelist for xid #%i...' % xid)
      for mesh_name in mesh_names:
        local_path = self.local_path_to_mesh(mesh_name, xid, split)
        if os.path.isfile(local_path) and not overwrite_if_present:
          continue
        local_dir = os.path.dirname(local_path)
        if not os.path.isdir(local_dir):
          os.makedirs(local_dir)
        remote_path = self.remote_path_to_mesh(mesh_name, xid, split)
        if file_util.exists(local_path):
          file_util.rm(local_path)
        file_util.cp(remote_path, local_path)

  def random_remote_mesh_names(self, count, split):
    """Generates random names of remote meshes."""
    candidates = self.all_remote_mesh_names(split)
    # random.shuffle(candidates)
    if count == -1 and candidates:
      return candidates
    chosen = candidates[:count]
    if len(chosen) != count:
      raise ValueError(
          'Not enough meshes are available. Requested %i but there are %i.' %
          (count, len(candidates)))
    return chosen

  def remote_path_to_mesh(self, mesh_name, xid, split):
    return os.path.join(
        self.remote_result_ckpt_dir(xid), str(self.ckpt_for_xid(xid, split)),
        split, self._mesh_relative_path(mesh_name))

  def local_path_to_mesh(self, mesh_name, xid, split):
    return os.path.join(
        self.local_result_ckpt_dir(xid), str(self.ckpt_for_xid(xid, split)),
        split, self._mesh_relative_path(mesh_name))

  def _mesh_relative_path(self, mesh_name):
    synset = mesh_name.split('|')[0]
    mesh_hash = mesh_name[len(synset)+1:]
    path = '%s/%s.ply' % (synset, mesh_hash)
    return path

  def local_qual_path(self):
    if self._local_qual_path:
      return self._local_qual_path
    path = os.path.join(self.local_root, self.experiment.experiment_name,
                        'images/')
    if not os.path.isdir(path):
      os.mkdir(path)
    self._local_qual_path = path
    return path


# TODO(kgenova) There should be only one model config class.
class ModelConfig(object):
  """A class to contain hyperparameters and other model attributes."""

  def __init__(self, hparams):
    self.hparams = hparams
    self.train = False
    self.eval = False
    self.inference = True
    self.input_data = None
    # TODO(kgenova) Maybe this should autopopulate to placeholders based on the
    # hparams. But it is a little dangerous to mess around with placeholders
    # since they require a graph. Maybe better to have a function for that?
    self.inputs = {'dataset': lambda: 0}

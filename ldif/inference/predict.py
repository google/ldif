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
"""Class to do trained model inference in beam."""

import importlib
import os
import struct
import subprocess as sp
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.datasets import preprocess
from ldif.datasets import shapenet
from ldif.inference import experiment as experiments
from ldif.inference import extract_mesh
from ldif.inference import metrics
from ldif.model import model as sdf_model
from ldif.representation import structured_implicit_function
from ldif.util import camera_util
from ldif.util import file_util
from ldif.util import gaps_util
from ldif.util import geom_util
from ldif.util import geom_util_np
from ldif.util import gpu_util
from ldif.util import path_util
from ldif.util import py_util
from ldif.util import sdf_util
from ldif.util import np_util

from ldif.util.file_util import log
# pylint: enable=g-bad-import-order

importlib.reload(extract_mesh)
importlib.reload(structured_implicit_function)
importlib.reload(sdf_model)
importlib.reload(geom_util)


class TrainedNetwork(object):
  """A base class for all networks trained in XManager."""

  def __init__(self, job, ckpt, use_gpu, **kwargs):  # pylint: disable=unused-argument
    self.job = job
    self.ckpt = ckpt
    self.graph = tf.Graph()
    self.use_gpu = use_gpu

  @classmethod
  def from_experiment(cls,
                      experiment,
                      xid,
                      ckpt_idx,
                      use_temp_ckpts=None,
                      overrides=None,
                      use_gpu=True,
                      **kwargs):
    """Instantiates a TrainedNetwork from an experiment object."""
    job = experiment.job_from_xmanager_id(xid, must_be_visible=True)
    if use_temp_ckpts is not None:
      job.set_use_temp_ckpts(use_temp_ckpts)
    if overrides is not None:
      for k, v in overrides.items():
        setattr(job.model_config.hparams, k, v)
    if ckpt_idx == 0:
      log.error('Please select a checkpoint and rerun. Valid checkpoints:')
      log.error(str(job.all_checkpoint_indices))
      return
    must_equal = ckpt_idx != -1
    ckpt = job.latest_checkpoint_before(ckpt_idx, must_equal=must_equal)
    log.info(f'Loading checkpoint {ckpt.abspath}')
    return cls(job, ckpt, use_gpu, **kwargs)

  @classmethod
  def from_modeldir(cls,
                    model_directory,
                    model_name,
                    experiment_name,
                    xid,
                    ckpt_idx,
                    overrides=None,
                    use_temp_ckpts=True,
                    use_gpu=True,
                    **kwargs):
    """Creates a TrainedModel from a model directory root and name."""
    experiment = experiments.Experiment(model_directory, model_name,
                                        experiment_name)
    return cls.from_experiment(experiment, xid, ckpt_idx, use_temp_ckpts,
                               overrides, use_gpu, **kwargs)

  @classmethod
  def from_identifiers(cls,
                       user,
                       model_name,
                       experiment_name,
                       xid,
                       ckpt_idx,
                       overrides=None,
                       use_temp_ckpts=None,
                       charged_user='viscam',
                       use_gpu=True,
                       **kwargs):
    """Creates a trained network from experiment identifiers."""
    raise ValueError('No longer supported.')

  def restore(self):
    """Creates a session with restored model variables."""
    with self.graph.as_default():
      if self.use_gpu:
        # For now these are disabled since it is difficult to work on
        # all GPUs.
        #allowable_frac = gpu_util.get_allowable_fraction_without(
        #    mem_to_reserve=1024 + 512, cuda_device_index=0)  # ~1GB
        #gpu_options = tf.GPUOptions(
        #    per_process_gpu_memory_fraction=allowable_frac)
        #config = tf.ConfigProto(gpu_options=gpu_options)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
      else:
        config = tf.ConfigProto(device_count={'GPU': 0})
      self.session = tf.Session(config=config)
      saver = tf.train.Saver()
      saver.restore(self.session, self.ckpt.abspath)


def conform_prediction(vector):
  """Forces an arbitrary vector to be a valid (D)SIF."""
  vector = vector.copy()
  if vector.shape[-1] not in [10, 42]:
    raise ValueError('Unimplemented.')
  consts, centers, radii_aa, radii_cov = np.split(
      vector[..., :10], [1, 4, 7], axis=-1)
  consts = np.minimum(consts, 0.0)
  radii_aa = np.maximum(radii_aa, 1e-9)
  radii_cov = np.clip(radii_cov, -np.pi / 4., np.pi / 4.)
  log.verbose(
      repr([
          x.shape
          for x in [consts, centers, radii_aa, radii_cov, vector[..., 10:]]
      ]))
  return np.concatenate(
      [consts, centers, radii_aa, radii_cov, vector[..., 10:]], axis=-1)


class SingleViewDepthEncoder(TrainedNetwork):
  """Maps from a single depth image (max-0) to a shape representation."""

  def __init__(self, job, ckpt, use_gpu, **kwargs):
    super(SingleViewDepthEncoder, self).__init__(job, ckpt, use_gpu, **kwargs)

    with self.graph.as_default():
      model_config = self.job.model_config
      model_config.inputs = shapenet.build_placeholder_interface(
          model_config, proto='ShapeNetOneImXyzPC')

      training_example = preprocess.preprocess(model_config)
      self.depth_input = model_config.inputs['dataset'].depth_render
      self.xyz_input = model_config.inputs['dataset'].xyz_render
      self.points_input = model_config.inputs['dataset'].surface_point_samples

      training_example = preprocess.preprocess(model_config)
      observation = sdf_model.Observation(model_config, training_example)
      imp_net = sdf_model.StructuredImplicitModel(model_config, 'imp_net')
      prediction = imp_net.forward(observation)
      structured_implicit = prediction.structured_implicit
      self.packed_vector = structured_implicit.vector
      self.restore()

  def run(self, depth, points, xyz):
    """Runs the network on the input data, returning a (D)SIF."""
    h, w = np.squeeze(depth).shape
    depth = np.reshape(depth, [1, h, w, 1])
    points = np.reshape(points, [1, 10000, 6])
    xyz = np.reshape(xyz, [1, h, w, 3])
    with self.graph.as_default():
      packed_vector = self.session.run(
          self.packed_vector,
          feed_dict={
              self.depth_input: depth,
              self.points_input: points,
              self.xyz_input: xyz
          })
      packed_vector = np.reshape(packed_vector,
                                 [self.job.model_config.hparams.sc, -1])
    return packed_vector

  def run_example(self, ex):
    return self.run(ex.max_depth_224[0, ...] * 1000.0,
                    ex.get_max_world_pts_from_idx(0), ex.max_world_xyz_224[0,
                                                                           ...])

  def run_example_bts(self, ex):
    return self.run(ex.bts_depth_224[0, ...] * 1000.0,
                    ex.get_bts_world_pts_from_idx(0), ex.bts_world_xyz_224[0,
                                                                           ...])


class DepthEncoder(TrainedNetwork):
  """Maps from a dodecahedron of depth images to shape elements."""

  def __init__(self, job, ckpt, use_gpu, **kwargs):
    super(DepthEncoder, self).__init__(job, ckpt, use_gpu, **kwargs)

    with self.graph.as_default():
      model_config = self.job.model_config
      model_config.hparams.bs = 1
      model_config.inputs = shapenet.build_placeholder_interface(model_config)

      training_example = preprocess.preprocess(model_config)
      self.depth_input = model_config.inputs['dataset'].depth_renders
      self.points_input = model_config.inputs['dataset'].surface_point_samples
      self.nss_input = model_config.inputs['dataset'].near_surface_samples

      training_example = preprocess.preprocess(model_config)
      if hasattr(training_example, '_tx'):
        self.tx = training_example._tx
      else:
        self.tx = None
      observation = sdf_model.Observation(model_config, training_example)
      imp_net = sdf_model.StructuredImplicitModel(model_config, 'imp_net')
      prediction = imp_net.forward(observation)
      structured_implicit = prediction.structured_implicit
      self.packed_vector = structured_implicit.vector
      # *phew* we have set up the graph... now we need to pull the weights.
      self.restore()

  def run(self, dodeca, points, nss=None):
    """Runs the network on the input data, returning a (D)SIF."""
    dodeca = np.reshape(dodeca, [1, 20, 224, 224, 1])
    points = np.reshape(points, [1, 10000, 6])
    with self.graph.as_default():
      feed_dict = {self.depth_input: dodeca, self.points_input: points}
      if nss is not None:
        feed_dict[self.nss_input] = np.reshape(nss, [1, 100000, 4])
      if self.tx is not None:
        packed_vector, tx = self.session.run([self.packed_vector, self.tx],
                                             feed_dict=feed_dict)
      else:
        packed_vector = self.session.run(
            self.packed_vector, feed_dict=feed_dict)
      packed_vector = np.reshape(packed_vector,
                                 [self.job.model_config.hparams.sc, -1])
    if self.tx is not None:
      return packed_vector, np.reshape(tx, [4, 4])
    return packed_vector

  def run_example(self, ex):
    return self.run(ex.depth_images, ex.precomputed_surface_samples_from_dodeca)


class Decoder(TrainedNetwork):
  """A SIF -> Mesh decoder."""

  def __init__(self, job, ckpt, use_gpu, **kwargs):
    super(Decoder, self).__init__(job, ckpt, use_gpu, **kwargs)

    with self.graph.as_default():
      self.sif_input = tf.placeholder(tf.float32, self.batched_vector_shape)
      # TODO(kgenova) Maybe the net should be handled entirely by the structured
      # implicit function? Although there is a difference between the network
      # that can give a result from a vector and a simple wrapper for models
      # that don't need variables. Maybe it's just intelligent about creating
      # the net only when really needed.
      if 'silence_implicits' in kwargs and kwargs['silence_implicits']:
        self.job.model_config.hparams.ipc = 'f'
        log.info('Silencing implicits.')
      net = sdf_model.StructuredImplicitModel(
          self.job.model_config, name='imp_net')
      structured_implicit = (
          structured_implicit_function.StructuredImplicit.from_packed_vector(
              self.job.model_config, self.sif_input, net))
      self.structured_implicit = structured_implicit

      self.block_res = 32
      self.native_point_count = self.block_res**3
      self.sample_locations_ph = tf.placeholder(
          tf.float32, shape=[self.block_res, self.block_res, self.block_res, 3])
      samples = tf.reshape(self.sample_locations_ph, [1, self.block_res**3, 3])
      predicted_alg, predicted_locals = structured_implicit.class_at_samples(
          samples, apply_class_transfer=False)
      predicted_class = sdf_util.apply_class_transfer(
          predicted_alg,
          self.job.model_config,
          soft_transfer=True,
          offset=self.job.model_config.hparams.lset)
      vol_shape = [self.block_res, self.block_res, self.block_res]
      self.predicted_alg_grid = tf.reshape(predicted_alg, vol_shape)
      self.predicted_class_grid = tf.reshape(predicted_class, vol_shape)
      effective_element_count = (
          structured_implicit_function.get_effective_element_count(
              self.job.model_config))
      self.local_decisions = tf.reshape(predicted_locals[0], [
          effective_element_count, self.block_res, self.block_res,
          self.block_res
      ])

      self.base_grid = np_util.make_coordinate_grid_3d(
          length=self.block_res,
          height=self.block_res,
          width=self.block_res,
          is_screen_space=False,
          is_homogeneous=False).astype(np.float32)

      self._world2local = structured_implicit.world2local

      self._use_inference_kernel = True

      # Influence samples
      self.true_sample_count = 10000
      self.generic_sample_ph = tf.placeholder(
          tf.float32, shape=[self.true_sample_count, 3])
      self.predicted_influences = structured_implicit.rbf_influence_at_samples(
          tf.expand_dims(self.generic_sample_ph, axis=0))

      # Optimizer stuff
      self.optimizer_pc = 5000
      self.optimizer_samples = tf.placeholder(
          tf.float32, shape=[self.optimizer_pc, 3])
      optimizer_samples = tf.reshape(self.optimizer_samples,
                                     [1, self.optimizer_pc, 3])
      self.predicted_class, _ = structured_implicit.class_at_samples(
          optimizer_samples)
      self.predicted_class = tf.reshape(self.predicted_class,
                                        [self.optimizer_pc, 1])
      self.target_class_ph = tf.placeholder(tf.float32, [self.optimizer_pc, 1])

      loss = 'crossentropy'
      if loss == 'crossentropy':
        clipped_pred = tf.clip_by_value(self.predicted_class, 1e-05, 1 - 1e-05)
        self.optimizer_elt_loss = tf.where(self.target_class_ph > 0.5,
                                           -tf.log(clipped_pred),
                                           -tf.log(1 - clipped_pred))
      elif loss == 'l1':
        self.optimizer_elt_loss = tf.abs(self.target_class_ph -
                                         self.predicted_class)
      elif loss == 'l2':
        self.optimizer_elt_loss = tf.square(self.target_class_ph -
                                            self.predicted_class)

      apply_where_agree = True
      if not apply_where_agree:
        gt_outside = self.target_class_ph > 0.5
        pred_outside = self.predicted_class > 0.5
        gt_inside = tf.logical_not(gt_outside)
        pred_inside = tf.logical_not(pred_outside)
        agree = tf.logical_or(
            tf.logical_and(gt_outside, pred_outside),
            tf.logical_and(gt_inside, pred_inside))
        self.optimizer_elt_loss = tf.where_v2(agree, 0.0,
                                              self.optimizer_elt_loss)

      self.optimizer_loss = tf.reduce_mean(self.optimizer_elt_loss)
      self.ldif_gradients = tf.gradients(self.optimizer_loss, self.sif_input)

      # TODO(kgenova) Currently disabled since it's in testing and hardcodes
      # some values.
      # self.coords_ph = tf.placeholder(tf.float32, shape=[3])
      # self.am_image_ph = tf.placeholder(tf.int32, shape=[224, 224])
      # pose_cam2world, pose_eye = self._spherical_to_4x4(self.coords_ph)
      # self.pose_error = self._evaluate_pose_error(pose_cam2world, pose_eye,
      #                                             self.am_image_ph)
      # self.pose3_gradients = tf.gradients(self.pose_error, self.coords_ph)
      try:
        self.restore()
      except ValueError:
        log.warning('No variables to restore or restoration otherwise failed.')

  @property
  def unbatched_vector_shape(self):
    shape_count = self.job.model_config.hparams.sc
    shape_size = structured_implicit_function.element_dof(self.job.model_config)
    return [shape_count, shape_size]

  @property
  def batched_vector_shape(self):
    return [1] + self.unbatched_vector_shape

  @property
  def use_inference_kernel(self):
    return self._use_inference_kernel

  @use_inference_kernel.setter
  def use_inference_kernel(self, should_use):
    self._use_inference_kernel = bool(should_use)

  # TODO(kgenova) The intermediate vector should really be its own class...
  def savetxt(self, sif_vector, path=None, version='v1'):
    """Saves a (D)SIF as ASCII text in the SIF file format.

    Args:
      sif_vector: A numpy array containing the ldif to write to disk. Has shape
        (element_count, element_length).
      path: A string containing the path to the file to write to, if provided.
        If none, no file is written.
      version: A string with the version identifier. Must equal 'v1'.

    Returns:
      A string encoding of the (D)SIF.
    """
    if version == 'v0':
      raise ValueError('SIF v0 files are no longer supported.')
    elif version == 'v1':
      s = self.encode_sif_v1(sif_vector)
    else:
      raise ValueError(f'Unrecognized SIF file format: {version}.')
    if path is not None:
      file_util.writetxt(path, s)
    return s

  def encode_sif_v1(self, sif_vector):
    """Encodes a ldif to a string, and optionally writes it to disk.

    A description of the file format:
    Line 1: SIF
    Line 2: Three ints separated by spaces. In order:
      1) The number of blobs.
      2) The version ID for the blob types. I added this to be safe since
         last time when we updated to add rotation it broke all the old txt
         files. For now it will always be zero, which means the following
         eleven explicit parameters will be given per blob (in order):
           1 constant. float.
           3 centers (XYZ). float.
           3 radii (XYZ diagonals). float.
           3 radii (roll-pitch-yaw rotations). float.
           1 symmetry ID type. int. For now it will be either 0 or 1:
               Zero: Not symmetric.
                One: Left-right (XY-plane) symmetry.
      3) The number of implicit parameters per blob. So it will likely
         be between 0-256.
    After the first two lines, there is a line for each blob.
     Each line will have the explicit parameters followed by the implicit
     parameters. They are space separated.

    Args:
     sif_vector: The SIF vector to encode as a np array. Has shape
       (element_count, element_length).

    Returns:
      A string encoding of v in the ldif v1 file format.
    """
    sif_vector = sif_vector.copy()
    shape_count = sif_vector.shape[-2]
    shape_len = sif_vector.shape[-1]
    if shape_len == 7:
      off_axis = np.zeros([shape_count, 3])
      sif_vector = np.concatenate([sif_vector, off_axis], axis=1)
      shape_len = 10
    explicit_len = 10
    implicit_len = shape_len - explicit_len
    sif_vector = np.reshape(sif_vector, [shape_count, shape_len])
    has_implicits = implicit_len > 0
    if not has_implicits:
      assert shape_len == 10
      implicit_len = 0
    sif_vector[:, 4:7] = np.sqrt(np.maximum(sif_vector[:, 4:7], 0))

    header = 'SIF\n%i %i %i\n' % (shape_count, 0, implicit_len)
    out = header
    for row_idx in range(shape_count):
      row = ' '.join(10 * ['%.9g']) % tuple(sif_vector[row_idx, :10].tolist())
      symmetry = int(row_idx < self.job.model_config.hparams.lyr)
      row += ' %i' % symmetry
      if has_implicits:
        implicit_params = ' '.join(implicit_len * ['%.9g']) % (
            tuple(sif_vector[row_idx, 10:].tolist()))
        row += ' ' + implicit_params
      row += '\n'
      out += row
    return out

  def render_ellipsoids(self, sif_vector):
    """Renders an ellipsoid image visualizing the (D)SIF RBFs."""
    with py_util.py2_temporary_directory() as d:
      qpath = d + '/q.txt'
      self.savetxt(sif_vector, qpath)
      impath = d + '/im.png'
      camera = ('1.0451 1.17901 0.630437 '
                '-0.614259 -0.695319 -0.373119 '
                '-0.547037 0.715996 -0.433705')
      with py_util.x11_server():
        cmd = '%s/qview %s -camera %s -image %s' % (path_util.gaps_path(),
                                                    qpath, camera, impath)
        sp.check_output(cmd, shell=True)
      im = file_util.read_image(impath)
    return im

  def interactive_viewer(self, sif_vector, mesh=None):
    """Opens a GAPS viewer that can display the SIF blobs alongside a mesh."""
    with py_util.py2_temporary_directory() as d:
      qpath = d + '/q.txt'
      self.savetxt(sif_vector, qpath)
      init_camera = ('1.0451 1.17901 0.630437 '
                     '-0.614259 -0.695319 -0.373119 '
                     '-0.547037 0.715996 -0.433705')
      mstr = ''
      if mesh is not None:
        mpath = d + '/m.ply'
        file_util.write_mesh(mpath, mesh)
        mstr = f' -input_mesh {mpath}'
      cmd = f'{path_util.gaps_path()}/qview {qpath} -camera {init_camera}{mstr}'
      sp.check_output(cmd, shell=True)

  def world2local(self, sif_vector):
    if sif_vector.shape[0] != 1:
      sif_vector = np.expand_dims(sif_vector, axis=0)
    m = self.session.run(
        self._world2local, feed_dict={self.sif_input: sif_vector})
    return m

  def interactive_mesh_viewer(self, sif_vector, resolution):
    """Opens up an OpenGL session viewing the mesh defined by the SIF/LDIF."""
    with py_util.py2_temporary_directory() as d:
      mpath = d + '/m.ply'
      m = self.extract_mesh(sif_vector, resolution)
      file_util.write_mesh(mpath, m)
      init_camera = ('1.0451 1.17901 0.630437 '
                     '-0.614259 -0.695319 -0.373119 '
                     '-0.547037 0.715996 -0.433705')
      cmd = '%s/mshview %s -camera %s' % (path_util.gaps_path(), mpath,
                                          init_camera)
      sp.check_output(cmd, shell=True)

  def interactive_gridview(self, sif_vector, resolution, extent=0.75):
    volume = self._grid_eval(
        sif_vector, resolution, extent, extract_parts=False, world2local=None)
    return gaps_util.grdview(volume)

  def _spherical_to_4x4(self, coords):
    """Turns spherical coords into a 4x4 affine transformation matrix."""
    r = coords[0]
    theta = coords[1]
    phi = coords[2]
    st = tf.sin(theta)
    x = r * st * tf.cos(phi)
    y = r * st * tf.sin(phi)
    z = r * tf.cos(theta)
    eye = tf.stack([x, y, z], axis=0)
    eye = tf.reshape(eye, [1, 3])
    center = tf.zeros([1, 3], dtype=tf.float32)
    world_up = tf.constant([[0., 1., 0.]], dtype=tf.float32)
    world2cam = camera_util.look_at(eye, center, world_up)
    cam2world = tf.linalg.inv(world2cam)
    cam2world = tf.constant(
        [[-9.9398971e-01, 2.7342862e-03, -4.7837296e-03, 1.4993416e-04],
         [1.6200442e-09, 8.6298174e-01, 4.9326313e-01, 7.1943283e-01],
         [5.5100261e-03, 4.9325553e-01, -8.6296844e-01, -1.2277470e+00],
         [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]],
        dtype=tf.float32)
    return tf.reshape(cam2world, [4, 4]), eye

  def _evaluate_pose_error(self, cam2world, eye, am_image):
    """Evaluates the error of an estimated 4x4 pose matrix."""
    # TODO(kgenova) Thisis a hack that only workds for 3d-r2n2
    ray_directions = gaps_util.gaps_depth_image_to_cam_image(
        np.ones((224, 224)), xfov=0.422204).astype(np.float32)
    tc = 15
    t_vals = tf.constant(np.arange(0.75, 2.25, .1), dtype=tf.float32)
    t_vals = tf.reshape(t_vals, [1, tc, 1])
    ray_count = int(np.prod(ray_directions.shape[:-1]))
    ray_directions = tf.reshape(ray_directions, [ray_count, 1, 3])
    eye = tf.reshape(eye, [1, 1, 3])
    cam_rays = ray_directions * t_vals + eye
    world_pts = geom_util.apply_4x4(
        cam_rays, cam2world, are_points=True, batch_rank=0, sample_rank=2)
    world_pts = tf.reshape(world_pts, [1, ray_count * tc, 3])
    self.cam_3dof_pts = world_pts
    world_rbfs = self.structured_implicit.rbf_influence_at_samples(world_pts)
    eec = world_rbfs.get_shape().as_list()[-1]
    assert len(am_image.get_shape().as_list()) == 2
    is_bg = tf.reshape(
        tf.logical_not(tf.equal(am_image, eec)), [1, ray_count, 1])
    am_image = tf.tile(tf.expand_dims(am_image, axis=-1), [1, 1, tc])
    flat_am = tf.reshape(am_image, [ray_count * tc, 1])
    flat_am = tf.where_v2(tf.equal(flat_am, 45), 0, flat_am)
    world_rbfs = tf.reshape(world_rbfs, [ray_count * tc, 45])

    max_val = tf.gather(world_rbfs, flat_am, batch_dims=1)
    max_val = tf.reshape(max_val, [1, ray_count, tc])
    max_val = tf.reduce_max(max_val, axis=-1)
    is_bg_mult = tf.cast(is_bg, dtype=tf.float32)
    max_val = is_bg_mult * max_val
    error = -1.0 * tf.reduce_sum(max_val)
    return error

  def optimize_3dof_pose(self, sif_vector, am_image, e, step_count=10, lr=1e-6):
    """Tries to fit a pose given a SIF in 3D and a SIF segmentation image."""
    if len(sif_vector.shape) == 2:
      sif_vector = np.expand_dims(sif_vector, axis=0)
    # Now rays is an array of shape [h, w, 3]. The origin is currently [0,0,0]
    # because the rays are in camera space (for now).
    lr = np.array([0.0, lr, lr], dtype=np.float32)
    # Just worry about a single step for now:
    # The pose is 3-dof: distance, phi, theta.
    coords = np.array([0.812717413913 / 1.75, 0.0, 0.0], dtype=np.float32)
    # cam2world, eye = self._spherical_to_4x4(coords)
    for i in range(step_count):
      log.verbose('Step %i: (%0.4f, %0.4f, %0.4f)' %
                  (i, coords[0], coords[1], coords[2]))
      grad, err, pts = self.session.run(
          [self.pose3_gradients, self.pose_error, self.cam_3dof_pts],
          feed_dict={
              self.am_image_ph: am_image,
              self.sif_input: sif_vector,
              self.coords_ph: coords
          })
      grad = grad[0]
      log.verbose('Error: %0.2f' % err)
      log.verbose('grad: %s' % repr(grad))
      log.verbose('pts.shape: ', repr(pts.shape))
      assert len(grad.shape) == 1
      assert grad.shape[0] == 3
      update = lr * grad
      log.verbose('Update: ', str(update))

      gaps_util.ptsview(pts, mesh=e.v1_gt_mesh)

      coords = coords - lr * grad
    return coords

  def optimize_to_gt(self,
                     sif_vector,
                     example,
                     step_count=1,
                     lr=0.01,
                     vis=0,
                     verbosity=0,
                     target='all',
                     samps='nss'):
    """Iteratively optimizes a SIF or LDIF to fit ground truth in/out values."""
    if samps == 'nss':
      all_samples = example.near_surface_samples.copy()
      np.random.shuffle(all_samples)
    elif samps == 'uni':
      all_samples = example.uniform_samples.copy()
    elif samps == 'nssuni':
      all_samples = np.concatenate(
          [example.near_surface_samples, example.uniform_samples], axis=0)
    elif samps == 'dodeca':
      depth_ims = example.depth_images / 1000.0
      all_samples = geom_util.depth_dodeca_to_samples(depth_ims)
    elif samps == 'depth':
      depth_idx = 1  # TODO(kgenova) Make this the one in the observation.
      depth_ims = example.depth_images / 1000.0
      depth_im = depth_ims[0, depth_idx, :, :, :]
      cam2world = geom_util.get_dodeca_camera_to_worlds()[depth_idx, :, :]
      assert depth_im.shape[0] == 224
      assert cam2world.shape[0] == 4
      log.verbose('Depth im shape: ', depth_im.shape)
      all_samples = geom_util.depth_image_to_samples(depth_im, cam2world)

    if verbosity >= 2:
      gaps_util.ptsview(all_samples[..., :], self.extract_mesh(sif_vector, 128))

    np.random.shuffle(all_samples)
    cl = all_samples[:, 3]
    all_samples[cl < 0, 3] = 0
    all_samples[cl > 0, 3] = 1

    samples, gt_class = np.split(all_samples, [3], axis=-1)
    samples = samples[:self.optimizer_pc, :]
    gt_class = gt_class[:self.optimizer_pc, :]

    def print_sat_count(vec):
      """Prints the number of contraints that are satisfied and the total."""
      pred = self.class_at_samples(vec, np.reshape(samples, [-1, 3]))

      pred_is_out = pred > 0.5
      gt_is_out = gt_class > 0.5
      log.verbose(pred_is_out.shape, gt_is_out.shape)
      agree = np.logical_or(
          np.logical_and(pred_is_out, gt_is_out),
          np.logical_and(
              np.logical_not(pred_is_out), np.logical_not(gt_is_out)))
      sat_count = np.count_nonzero(agree)
      log.info('%i/%i constraints are satisfied.' %
               (sat_count, self.optimizer_pc))

    if verbosity >= 1:
      log.info('Beginning optimization.')
      print_sat_count(sif_vector)
    assert gt_class.shape[-1] == 1
    sif_vector = sif_vector.copy()
    sif_vector = np.expand_dims(sif_vector, axis=0)
    cur_vector = sif_vector.copy()
    ret_best = False
    if ret_best:
      min_loss = np.inf
      best_vec = cur_vector.copy()
    momentum = 0.9
    velocity = np.zeros_like(cur_vector)
    cur_batch_idx = 0
    for i in range(step_count):

      batch_start = cur_batch_idx
      batch_end = cur_batch_idx + self.optimizer_pc
      if batch_end > all_samples.shape[0]:
        np.random.shuffle(all_samples)
        batch_start = 0
        batch_end = self.optimizer_pc
        cur_batch_idx = 0
      batch_all_samples = all_samples[batch_start:batch_end, :]
      cur_batch_idx += self.optimizer_pc
      batch_samples, batch_gt_class = np.split(batch_all_samples, [3], axis=-1)

      grad = self.session.run(
          self.ldif_gradients,
          feed_dict={
              self.target_class_ph: batch_gt_class,
              self.sif_input: cur_vector,
              self.optimizer_samples: batch_samples
          })[0]
      vis_this_time = vis >= 2 or (vis >= 1 and (i == 0 or i == step_count - 1))
      print_this_time = verbosity >= 2 or (verbosity >= 1 and not i % 1000)
      if vis_this_time or print_this_time:
        loss = self.session.run(
            self.optimizer_elt_loss,
            feed_dict={
                self.target_class_ph: batch_gt_class,
                self.sif_input: cur_vector,
                self.optimizer_samples: batch_samples
            })
        if ret_best:
          lsum = np.sum(loss)
          if lsum < min_loss:
            min_loss = lsum
            best_vec = cur_vector.copy()
        # Assuming the loss is zero if a constraint is satisfied:
        is_sat = self.optimizer_pc - np.count_nonzero(loss)
        if print_this_time:
          log.info('Step %i: Total loss: %s. Constraints %i/%i' %
                   (i, repr(np.sum(loss)), is_sat, self.optimizer_pc))
        if vis_this_time:
          self.vis_loss(
              cur_vector,
              gt_at_loss=gt_class,
              loss=loss,
              loss_positions=samples)
      if target == 'all-eq':
        mults = 42 * [1]
      elif target == 'all':
        mults = [0.001] + 3 * [0.001] + 6 * [0.0000001] + 32 * [50]
      elif target == 'centers':
        mults = [0.000] + 3 * [0.001] + 6 * [0.0000000] + 32 * [0]
      elif target == 'radii':
        mults = [0.000] + 3 * [0.000] + 6 * [0.0000001] + 32 * [0]
      elif target == 'features':
        mults = [0.000] + 3 * [0.000] + 6 * [0.0000000] + 32 * [50]
      elif target == 'constants':
        mults = [0.001] + 3 * [0.000] + 6 * [0.0000000] + 32 * [0]
      else:
        assert False
      mults = np.array(mults).reshape([1, 1, 42])
      velocity = momentum * velocity + mults * lr * grad
      cur_vector = cur_vector - velocity
    if verbosity >= 1:
      log.info('Finished optimization.')
      print_sat_count(cur_vector)
    if ret_best:
      cur_vector = best_vec
    return np.reshape(cur_vector, self.unbatched_vector_shape)

  def vis_loss(self, sif_vector, gt_at_loss, loss, loss_positions):
    """Visualizes the loss mid-optimization."""
    loss = np.reshape(loss, [-1, 1])
    gt_at_loss = np.reshape(gt_at_loss, [-1, 1])
    assert gt_at_loss.shape[0] == loss.shape[0]
    loss[gt_at_loss <= 0.5] = -loss[gt_at_loss <= 0.5]
    loss_positions = np.reshape(loss_positions, [-1, 3])
    arr = np.concatenate([loss_positions, loss], axis=1)
    with py_util.py2_temporary_directory() as d:
      sdf_path = f'{d}/a.sdf'
      with file_util.open_file(sdf_path, 'wb') as f:
        arr = arr.astype(np.float32)
        arr.tofile(f)
      m = self.extract_mesh(sif_vector, resolution=128)
      m_path = f'{d}/m.ply'
      file_util.write_mesh(m_path, m)
      init_camera = ('1.0451 1.17901 0.630437 '
                     '-0.614259 -0.695319 -0.373119 '
                     '-0.547037 0.715996 -0.433705')
      cmd = '%s/ptsview %s %s -camera %s' % (path_util.gaps_path(), sdf_path,
                                             m_path, init_camera)
      sp.check_output(cmd, shell=True)

  def _grid_eval_cuda(self, sif_vector, resolution, extent):
    """Evaluates a SIF/LDIF densely on a voxel grid."""
    log.verbose('Using custom CUDA kernel for evaluation.')

    # First step: Get the path where the serialized occnet should be.
    # The serialized occnet should be at whatever the checkpoint path is,
    # but replace model.ckpt-[idx] with serialized-occnet-[idx].occnet
    checkpoint_path = self.ckpt.abspath
    log.info(f'Using checkpoint {checkpoint_path} to write OccNet file.')
    assert 'model.ckpt-' in checkpoint_path
    occnet_path = checkpoint_path.replace('model.ckpt-', 'serialized-occnet-')
    occnet_path = occnet_path + '.occnet'
    # Second step: If it isn't there, write it to disk.
    if not os.path.isfile(occnet_path):
      assert os.path.isdir(os.path.dirname(occnet_path))
      if self.job.model_config.hparams.ipe == 't':
        self.write_occnet_file(occnet_path)
      else:
        occnet_path = path_util.get_path_to_ldif_root(
        ) + '/ldif2mesh/extracted.occnet'
    # Third step: open a temporary directory, and write the embedding.
    #   Make sure that the temp directories are deleted afterwards.
    with py_util.py2_temporary_directory() as d:
      rep_path = f'{d}/ldif.txt'
      self.savetxt(sif_vector, rep_path)

      # Pick the path to the output grd file:
      grd_path = f'{d}/grid.grd'

      # Fourth step: Get the path to the kernel
      kernel_path = os.path.join(path_util.get_path_to_ldif_root(),
                                 'ldif2mesh/ldif2mesh')
      if not os.path.isfile(kernel_path):
        raise ValueError(
            f'There is no compiled CUDA executable at {kernel_path}.')

      cmd = (f'CUDA_VISIBLE_DEVICES=0 {kernel_path} {rep_path} {occnet_path} '
             f'{grd_path} -resolution {resolution}')
      log.verbose(f'Executing command {cmd}')

      # TODO(kgenova) Support extent as a flag
      if extent != 0.75:
        raise ValueError(
            'Currently only 0.75 extent is supported on the '
            'custom kernel. Please set use_inference_kernel to false for an'
            f' extent of {extent}.')
      # Fifth step: Invoke the kernel.
      try:
        cmd_result = sp.check_output(cmd, shell=True)
        log.info(cmd_result.decode('utf-8').replace('\n', ''))
      except sp.CalledProcessError as e:
        if 'out of memory' in e.output.decode('utf-8'):
          raise ValueError(
              'The GPU does not have enough free memory left for the'
              ' inference kernel. Please reduce the fraction'
              ' reserved by tensorflow.')
        elif 'no kernel image is available' in e.output.decode('utf-8'):
          raise ValueError(
              'It appears that the CUDA kernel was not built to your '
              'gpu\'s architecture. Hopefully this is an easy fix. '
              'Please go to developer.nvidia.com/cuda-gpus, and find '
              'your gpu from the list. Then, modify ./build_kernel.sh '
              'by adding compute_XX and sm_XX for whatever your GPU '
              'compute capability is according to the website. For '
              'example, a 2080 Ti would use compute_75 and sm_75. '
              'Note that if your card supports below 35, it likely '
              'will fail to compile using this method. If you are '
              'seeing this error, please feel free to open up an issue '
              'and report it. We would like to support as many gpus as '
              'possible.')
        else:
          raise ValueError(f'Unrecognized error code {e.returncode} occurred'
                           f' during inference kernel evaluation: {e.output}')

      # Seventh step: Read the grid file.
      _, grd = file_util.read_grd(grd_path)
    # Eighth step: Verify the grid shape and return the grid.
    log.verbose(f'The output CUDA grid has shape {grd.shape}.')
    # gaps_util.grdview(grd)
    return grd

  def _grid_eval(self,
                 sif_vector,
                 resolution,
                 extent,
                 extract_parts,
                 world2local=None):
    """Evalutes the LDIF/SIF on a grid."""
    log.verbose('Evaluating SDF grid for mesh.')
    if self.use_inference_kernel and not extract_parts:
      return self._grid_eval_cuda(sif_vector, resolution, extent)
    if extract_parts or world2local:
      log.warning('Part extraction and world2local are not supported with the'
                  ' custom kernel.')
    log.warning('Using pure tensorflow for grid evaluation, this will be slow.')
    t = time.time()
    sif_vector = np.reshape(sif_vector, self.batched_vector_shape)
    assert not resolution % self.block_res
    block_count = resolution // self.block_res
    block_size = (2.0 * extent) / block_count
    l_block = []
    i = 0
    dim_offset = 1 if extract_parts else 0
    grid = self.local_decisions if extract_parts else self.predicted_alg_grid
    for li in range(block_count):
      l_min = -extent + (li) * block_size - 0.5 / resolution
      h_block = []
      for hi in range(block_count):
        h_min = -extent + (hi) * block_size - 0.5 / resolution
        w_block = []
        for wi in range(block_count):
          w_min = -extent + (wi) * block_size - 0.5 / resolution
          offset = np.reshape(
              np.array([w_min, l_min, h_min], dtype=np.float32), [1, 1, 1, 3])
          sample_locations = block_size * self.base_grid + offset
          if world2local is not None:
            sample_locations = geom_util_np.apply_4x4(
                sample_locations, world2local, are_points=True)
          grid_out_np = self.session.run(
              grid,
              feed_dict={
                  self.sif_input: sif_vector,
                  self.sample_locations_ph: sample_locations
              })
          i += 1
          w_block.append(grid_out_np)
        h_block.append(np.concatenate(w_block, axis=2 + dim_offset))
      l_block.append(np.concatenate(h_block, axis=0 + dim_offset))
    grid_out = np.concatenate(l_block, axis=1 + dim_offset)
    # log.verbose(f'Grid extent: {np.min(grid_out)}, {np.max(grid_out)}')
    # grid_out -= 0.5
    grid_out_time = time.time()
    log.verbose(f'Grid Eval Time: {grid_out_time - t}')
    return grid_out

  def extract_mesh(self,
                   sif_vectors,
                   resolution=128,
                   extent=0.75,
                   return_success=False,
                   world2local=None):
    """Extracts a mesh that is the sum of one or more SIF meshes."""
    extract_start_time = time.time()
    if isinstance(sif_vectors, list):
      volumes = []
      if world2local is not None:
        assert isinstance(world2local, list)
      for i, v in enumerate(sif_vectors):
        volumes.append(
            self._grid_eval(
                v,
                resolution,
                extent,
                extract_parts=False,
                world2local=world2local[i]
                if world2local is not None else None))
      volume = np.sum(volumes, axis=0)
    else:
      volume = self._grid_eval(
          sif_vectors,
          resolution,
          extent,
          extract_parts=False,
          world2local=world2local)
    grid_out_time = time.time()
    log.verbose(f'Grid eval time: {grid_out_time - extract_start_time}')
    had_crossing, mesh = extract_mesh.marching_cubes(volume, extent)
    if not had_crossing:
      log.warning('Warning: Marching Cubes found no surface.')
    mesh.marching_cubes_successful = had_crossing
    done_time = time.time()
    log.verbose(f'MCubes Time: {done_time - grid_out_time}')
    if return_success:
      return mesh, had_crossing
    return mesh

  def extract_part_meshes(self, sif_vector, resolution, extent=0.75):
    elt_volume = self._grid_eval(
        sif_vector, resolution, extent, extract_parts=True, world2local=None)
    local_meshes = []
    for i in range(self.job.model_config.hparams.sc):
      had_crossing, mesh_i = extract_mesh.marching_cubes(
          elt_volume[i, ...], extent)
      mesh_i.marching_cubes_successful = had_crossing
      local_meshes.append(mesh_i)
    return local_meshes

  def _chunk_sample_eval(self, samples, query_fun, chunk_size):
    """Evaluates a set of query locations chunk by chunk to avoid OOM issues."""
    # Note- this code will have strange behavior if there is randomness during
    # decoding, because it chunks the decoding up into multiple calls.
    assert len(samples.shape) == 2
    point_count = samples.shape[0]
    if point_count == chunk_size:
      chunks = [samples]
    else:
      pad_len = chunk_size - (point_count % chunk_size)
      if pad_len:
        samples = np.pad(samples, ((0, pad_len), (0, 0)), 'constant')
      assert not (point_count + pad_len) % chunk_size
      chunk_count = (point_count + pad_len) // chunk_size
      chunks = np.split(samples, chunk_count, axis=0)
    out = []
    for chunk in chunks:
      out_i = query_fun(chunk)
      assert len(out_i.shape) == 2
      assert out_i.shape[0] == chunk_size
      out.append(out_i)
    return np.concatenate(out, axis=0)[:point_count, :]

  def iou(self, sif_vector, example):
    samps = example.uniform_samples[:, :3]
    gt_is_inside = example.uniform_samples[:, 3:4] < 0.0
    pred_is_inside = self.class_at_samples(sif_vector, samps) < 0.5
    result = metrics.point_iou(pred_is_inside, gt_is_inside)
    return result

  def class_at_samples(self, sif_vector, samples):
    """Determines whether input xyz locations are inside or outside the shape.

    Args:
      sif_vector: A numpy array containing the LDIF/SIF to evaluate. Has shape
        (element_count, element_length).
      samples: A numpy array containing samples in the LDIF/SIF frame. Has shape
        (sample_count, 3).

    Returns:
      A numpy array with shape (sample_count, 1). A float that is positive
      outside the LDIF/SIF, and negative inside.
    """
    sif_vector = np.reshape(sif_vector, self.batched_vector_shape)

    def query(sample_chunk):
      chunk_grid = sample_chunk.reshape(
          [self.block_res, self.block_res, self.block_res, 3])
      classes = self.session.run(
          self.predicted_class_grid,
          feed_dict={
              self.sif_input: sif_vector,
              self.sample_locations_ph: chunk_grid
          })
      classes = classes.reshape([self.block_res**3, 1])
      return classes

    return self._chunk_sample_eval(samples, query, self.block_res**3)

  def rbf_influence_at_samples(self, sif_vector, samples):
    """Evalutes the influence of each RBF in the SIF/LDIF at each sample.

    Args:
      sif_vector: A numpy array containing the ldif to evaluate. Has shape
        (element_count, element_length).
      samples: A numpy array containing the samples in the ldif frame. Has shape
        (sample_count, 3).

    Returns:
      A numpy array with shape (sample_count, effective_element_count). The
      RBF weight of each effective element at each sample point. The 'effective'
      element count may be higher than the element count, depending on the
      symmetry settings of the ldif. In the case where a ldif is partially
      symmetric, then some elements have multiple RBF weights- their main weight
      (given first) and the weight associated with the shadow element(s)
      transformed by their symmetry matrix. See structured_implicit_function.py
      for a mapping from element indices to equivalent classes. Regardless of
      additional 'effective' elements, the first RBF weights correspond to the
      'real' elements with no symmetry transforms applied, in order.
    """
    # TODO(kgenova) It's a bit clunky to make the user refer to a different
    # python file to get symmetry equivalence classes. Maybe that mapping should
    # be returned as needed.
    sif_vector = np.reshape(sif_vector, self.batched_vector_shape)

    def query(sample_chunk):
      chunk_in = sample_chunk.reshape([self.true_sample_count, 3])
      influences = self.session.run(
          self.predicted_influences,
          feed_dict={
              self.generic_sample_ph: chunk_in,
              self.sif_input: sif_vector
          })
      return np.squeeze(influences)

    return self._chunk_sample_eval(samples, query, self.true_sample_count)

  def write_occnet_file(self, path):
    """Serializes an occnet network and writes it to disk."""
    f = file_util.open_file(path, 'wb')
    # Get the weight tensors associated with the occnet:
    with self.graph.as_default():
      all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      occnet_vars = contrib_framework.filter_variables(
          all_vars, include_patterns=['eval_implicit_parameters'])

    # Extract all the model weights as numpy values:
    model = {}
    for v in occnet_vars:
      value = self.session.run(v)
      log.verbose(f'{v.name}: {value.shape}')
      assert v.name not in model
      model[v.name] = value

    # Serialize them into a single file:
    def write_header(base_scope):
      # Write the shape so the number of occnet resnet layers and their sizes
      # are known.
      num_resnet_layers = 1
      # Writes all arrays in row-major order.
      dim = model[base_scope +
                  'sample_resize_fc/fully_connected/weights:0'].shape[1]
      log.verbose(f'Dimensionality is {dim}')
      f.write(struct.pack('ii', num_resnet_layers, dim))

    def write_fc_layer(layer_scope):
      weights = model[layer_scope + '/fully_connected/weights:0']
      biases = model[layer_scope + '/fully_connected/biases:0']
      log.verbose(f'FC layer shapes: {weights.shape}, {biases.shape}')
      f.write(weights.astype('f').tostring())
      f.write(biases.astype('f').tostring())

    def write_cbn_layer(layer_scope):
      write_fc_layer(layer_scope + '/beta_fc')
      write_fc_layer(layer_scope + '/gamma_fc')
      running_mean = float(model[layer_scope + '/running_mean:0'])
      running_var = float(model[layer_scope + '/running_variance:0'])
      log.verbose(f'Running mean, variance: {running_mean}, {running_var}')
      f.write(struct.pack('ff', running_mean, running_var))

    def write_input_layer(layer_scope):
      weights = model[layer_scope + '/fully_connected/weights:0']
      biases = model[layer_scope + '/fully_connected/biases:0']
      log.verbose(f'Input FC layer shapes: {weights.shape}, {biases.shape}')
      f.write(weights.astype('f').tostring())
      f.write(biases.astype('f').tostring())

    def write_activation_layer(layer_scope):
      weights = model[layer_scope + '/fully_connected/weights:0']
      bias = float(model[layer_scope + '/fully_connected/biases:0'])
      log.verbose(f'Final FC layer shape and bias: {weights.shape}, {bias}')
      f.write(weights.astype('f').tostring())
      f.write(struct.pack('f', bias))

    base = 'imp_net/eval_implicit_parameters/all_elements/OccNet/'
    write_header(base)
    write_input_layer(base + 'sample_resize_fc')
    write_cbn_layer(base + 'fc_resnet_layer_0/cbn_1')
    write_fc_layer(base + 'fc_resnet_layer_0/fc_1')
    write_cbn_layer(base + 'fc_resnet_layer_0/cbn_2')
    write_fc_layer(base + 'fc_resnet_layer_0/fc_2')
    write_cbn_layer(base + 'final_cbn')
    write_activation_layer(base + 'final_activation')
    f.close()

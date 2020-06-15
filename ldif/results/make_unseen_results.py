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
"""Evaluates a network on a dataset and writes the result in an npy file."""

import os
import pickle

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.datasets import shapenet_numpy
from ldif.inference import example as examples
from ldif.inference import predict
from ldif.results import build_inputs
from ldif.results import results_pb2
from ldif.util import beam_util
from ldif.util import file_util
from ldif.util import mesh_util
from ldif.util import py_util
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order


FLAGS = flags.FLAGS

flags.DEFINE_string('split', 'test',
                    'The dataset split on which to run inference.')

flags.DEFINE_string('dataset', 'shapenet-unseen',
                    'The input data+GT pair to run on.')

flags.DEFINE_string('category', 'unseen', 'The category to run on.')

flags.DEFINE_integer('instance_count', '-1',
                     'The number of instances to run on. -1 means all.')

flags.DEFINE_string('experiment_name', None,
                    'The name of the experiment to evaluate.')
flags.mark_flag_as_required('experiment_name')

flags.DEFINE_string('xids', '', 'XManager ids to evaluate.')

flags.DEFINE_string('model_name', 'sif-transcoder',
                    'The model name (usually sif-transcoder).')

flags.DEFINE_string(
    'out_dir',
    '',
    'The root directory for the out path.')

flags.DEFINE_integer('resolution', 256, 'The marching cubes resolution to use.')

flags.DEFINE_integer('ckpt', 0, 'The checkpoint index.')

flags.DEFINE_string(
    'xid_to_ckpt', '',
    'A (xid1, ckpt1, xid2, ckpt2, ...) comma separated list of '
    'approximate checkpoint inds.')

flags.DEFINE_boolean(
    'allow_fails', None,
    'Whether to allow evaluation to fail on any mesh. If true, a log will be'
    ' generated indicating which examples failed and what the error was.')
flags.mark_flag_as_required('allow_fails')


def get_serialized_inputs_for_elt(elt, example):
  """Serializes the example to a string for processing."""
  # TODO(kgenova) Maybe these should live in the predict.py class
  if 'shapenet' in elt['dataset']:
    if 'unseen' in elt['dataset']:
      return []
    if 'autoencoder' in elt['dataset']:
      depth = pickle.dumps(example.depth_images)
      pts = pickle.dumps(example.surface_samples_from_dodeca)
      return [depth, pts]
    elif 'depth' in elt['dataset']:
      depth = pickle.dumps(example.max_depth_224[0, ...] * 1000.0)
      pts = pickle.dumps(example.get_max_world_pts_from_idx(0))
      xyz = pickle.dumps(example.max_world_xyz_224[0, ...])
      return [depth, pts, xyz]
    elif 'rgb' in elt['dataset']:
      rgb = pickle.dumps(example.r2n2_images[0, ...])
      xfov = pickle.dumps(example.r2n2_xfov[0])
      cam2world = pickle.dumps(example.r2n2_cam2world[0, ...])
      return [rgb, xfov, cam2world]
  raise ValueError('Unimplemented.')


def make_example(elt):
  """Makes a python object with the example dataset items."""
  if 'shapenet' in elt['dataset'] and 'unseen' not in elt['dataset']:
    return examples.InferenceExample.from_elt(elt)
  elif 'unseen' in elt['dataset']:
    obj = lambda: 0
    # TODO(ldif-user) Set up the input key-value store
    kv_store = None
    k = elt['mesh_identifier'].strip()
    s = kv_store[k]
    p = shapenet_numpy.ShapeNetNSSDodecaSparseLRGMediumSlimPC.FromString(s)
    obj.depth_images = p.depth_renders
    obj.surface_samples_from_dodeca = p.surface_point_samples
    # TODO(ldif-user) Set up the path to the transformation:
    tx_path = 'ROOT_DIR/%s/occnet_to_gaps.txt' % k
    obj.occnet_to_gaps = file_util.read_txt_to_np(tx_path).reshape([4, 4])
    obj.gaps_to_occnet = np.linalg.inv(obj.occnet_to_gaps)
    # TODO(ldif-user) Set up the path to the normalized gt mesh.
    nrm_gt_mesh_path = 'ROOT_DIR/%s/model_normalized.ply' % k
    obj.normalized_gt_mesh = file_util.read_mesh(nrm_gt_mesh_path)
    obj.gt_mesh = obj.normalized_gt_mesh.copy()
    obj.gt_mesh.apply_transform(obj.gaps_to_occnet)
    return obj
  else:
    raise ValueError('Unimplemented.')


def make_encoder(elt):
  """Builds an appropriate encoder to encode the element."""
  if elt['dataset'] in ['shapenet-autoencoder', 'shapenet-unseen']:
    constructor = predict.DepthEncoder
  elif elt['dataset'] == 'shapenet-depth':
    constructor = predict.SingleViewDepthEncoder
  elif elt['dataset'] == 'shapenet-rgb':
    constructor = predict.RgbEncoder
  else:
    raise ValueError('Unimplemented: make_encoder')

  # TODO(ldif-user) Build the encoder.
  encoder = constructor(None)
  return encoder


def try_to(fun, elt, net=None):
  """Attempts a function, adding 'FAILED' as a key to elt if it fails."""
  if FLAGS.allow_fails:
    try:
      if net is not None:
        return fun(elt, net)
      else:
        return fun(elt)
    # pylint: disable=broad-except
    except Exception as e:
      # pylint: enable=broad-except
      msg = ('Failed on mesh %s in %s() with error %s' %
             (elt['mesh_identifier'], repr(fun), repr(e)))
      log.error(msg)
      elt['FAILED'] = msg
      return elt
  else:
    if net is not None:
      return fun(elt, net)
    else:
      return fun(elt)


def transform_mesh_to_gt_frame(elt, pred_mesh, extraction_successful, example):
  if 'shapenet' not in elt['dataset']:
    return pred_mesh
  if extraction_successful:
    pred_mesh = pred_mesh.copy()
    pred_mesh = pred_mesh.apply_transform(example.gaps_to_occnet)
    return pred_mesh
  else:
    # We just have a sphere, don't try to un-normalize it:
    return pred_mesh


class MapNetwork(beam.DoFn):
  """A class to apply a map function that requires initializing a network."""

  def __init__(self, map_fun, net_initializer):
    super(MapNetwork, self).__init__()
    self.map_fun = map_fun
    self.net_initializer = net_initializer

  def initialize(self, elt):
    if hasattr(self, 'net'):
      # TODO(kgenova) We could check here for a more sophisticated setup
      # that would allow for pcolls that mix elts requiring different networks.
      # Already initialized:
      return
    self.net = self.net_initializer(elt)

  def process(self, elt):
    self.initialize(elt)
    return [self.map_fun(elt, net=self.net)]


def _encode(elt, encoder):
  """Encodes an element, adding an 'embedding' field to the vector."""
  example = make_example(elt)
  # encoder = make_encoder(elt)
  elt['embedding'] = encoder.run_example(example)
  elt['inputs'] = get_serialized_inputs_for_elt(elt, example)
  return elt
encode = lambda elt, net: try_to(_encode, elt, net)


def make_decoder(elt):  # pylint:disable=unused-argument
  """Makes a decoder object for the elt."""
  # TODO(ldif-user) Make the decoder object
  decoder = None
  return decoder


def _decode(elt, decoder):
  """Adds the decoder-related attributes to the elt."""
  example = make_example(elt)
  sif_vector = elt['embedding']
  elt['representation'] = decoder.savetxt(sif_vector, path=None, version='v1')
  elt['gt_mesh'] = mesh_util.serialize(example.gt_mesh)
  mesh, succeeded = decoder.extract_mesh(
      sif_vector,
      resolution=FLAGS.resolution,
      extent=0.75,
      return_success=True)
  mesh = transform_mesh_to_gt_frame(elt, mesh, succeeded, example)
  elt['mesh'] = mesh_util.serialize(mesh)
  elt['extraction_success'] = succeeded
  return elt
decode = lambda elt, net: try_to(_decode, elt, net)


def make_proto(elt):
  """Builds a Results proto object from the elt."""
  proto = results_pb2.Results()
  proto.mesh_identifier = elt['mesh_identifier']
  proto.dataset = elt['dataset']
  proto.split = elt['split']
  proto.user = elt['user']
  proto.model_name = elt['model_name']
  proto.experiment_name = elt['experiment_name']
  proto.xid = elt['xid']
  proto.ckpt = elt['ckpt']
  proto.representation = elt['representation']
  proto.gt_mesh = elt['gt_mesh']
  proto.mesh = elt['mesh']
  proto.extraction_success = elt['extraction_success']
  for ipt in elt['inputs']:
    proto.inputs.append(ipt)
  return elt['mesh_identifier'], proto.SerializeToString()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Otherwise models that use tf hub will run out of local disk space:
  # TODO(ldif-user) Set the tf-hub cache directory:
  os.environ['TFHUB_CACHE_DIR'] = ('')

  tasks = build_inputs.generate_tasks(FLAGS)
  log.info('Tasks:')
  log.info(tasks)

  models = build_inputs.generate_models(FLAGS)
  log.info('Models:')
  log.info(models)

  fail_base = FLAGS.out_dir + '/fails'
  output_base = FLAGS.out_dir + '/results'

  # TODO(ldif-user): Set up your own pipeline runner
  with beam.Pipeline() as p:
    for model in models:
      log.info('Making pipeline for model %s' % repr(model))
      name = 'XID%i' % model['xid']
      inputs = py_util.merge_into(model, tasks)
      log.info('Inputs for model %s: %s' % (name, repr(inputs)))
      inputs = p | ('CreateInputs%s' % name) >> beam.Create(inputs)

      encoded = beam_util.map_and_report_failures(
          inputs,
          MapNetwork(encode, make_encoder),
          'Encode%s' % name,
          fail_base,
          applier=beam.ParDo)
      decoded = beam_util.map_and_report_failures(
          encoded,
          MapNetwork(decode, make_decoder),
          'Decode%s' % name,
          fail_base,
          applier=beam.ParDo)

      protos = decoded | 'MakeProto%s' % name >> beam.Map(make_proto)
      # TODO(ldif-user) Set file extension
      extension = None
      output_path = '%s-%s-%i.%s' % (output_base, name, model['ckpt'],
                                     extension)
      # TODO(ldif-user) Replace lambda x: None with a proto sink.
      _ = protos | 'Write%s' % name >> (lambda x: None)

  log.info('Pipeline Finished!')
  log.info('The output directory is: %s' % FLAGS.out_dir)


if __name__ == '__main__':
  app.run(main)

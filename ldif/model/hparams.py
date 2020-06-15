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
"""hyperparameters for the model."""

import pickle
import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import file_util
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order


rgb2q_hparams = tf.contrib.training.HParams(
    bs=32,  # The batch size.
    lr=5e-05,  # The learning rate.
    sc=25,  # The number of primitives (must match the dataset).
    sd=42,  # The length of each primitive
    ed=10,  # The length of the explicit component.
    fcc=2,  # The number of latent fully connected layers. >= 0.
    fcl=2048,  # The width of the latent fully connected layers.
    # ??
    ft='t',
    # How to apply input augmentation: 'f', 't', 'io'
    aug='f',
    # The encoder architecture. 'rn', 'sr50', 'vl', 'vlp':
    enc='rn',
    # The levelset of the reconstruction:
    lset=-0.07,
    # The distillation loss type: 'l1', 'l2', or'hu'
    l='l1',
    # [w]: The distillation loss weighting:
    # 'r': Relative based on [ilw].
    # 'u': Uniform 1.0
    # 'i': inverse weighting (basically bad)
    w='r',
    # Distillation supervision:
    # 'a': All
    # 'i': Just implicits
    # 'e': Just explicits
    sv='a',
    # The background: 'b' or 'w' (black/white) with or without 's' for smooth:
    bg='ws',
    # The distillation loss weight:
    dlw=1.0,
    # The relative weight of implicits within the distillation loss:
    ilw=1.0,
    # The relative weight of explicits within the distillation loss:
    elw=1.0,
    # The argmax segmentation task loss weight:
    slw=1.0,
    # The depth task loss weight:
    tlw=1.0,
    # The xyz task loss weight:
    xlw=1.0,
    # The set of extra tasks:
    # 'd': Predict depth
    # 'x': Predict XYZ
    # 'a': Predict the argmax image.
    # 's': Predict a segmentation image.
    # 'n': Predict a normals image.
    xt='a',
    # Whether to apply whitening:
    wh='t',
    # The reconstruction loss weight:
    rlw=1.0,
    # The operating resolution (excluding pretrained networks).
    res=137,
    # The decoder architecture:
    dec='dl',
    # Whether to enable tiling to three channels for pretrained input support:
    et='f',
    # The type of image to use for the input:
    im='rgb',
    # Whether to predict the blobs from the extra prediction.
    eil='f',
    # Whether the secondary encoder is pretrained (if there is one)
    spt='t',
    # The secondary encoder architecture.
    sec='epr50',
    # Whether the depth prediction should be transformed to XYZ before being
    # given to the secondary encoder:
    txd='t',
    # Whether to use the predicted image 't' or the GT image 'f' in the inline-
    # encoder:
    up='t',
    # Whether to apply the depth prediction loss in xyz space:
    rdx='f',
    # Whether each task gets its own decoder:
    st='f',
    # [lsg]: What kind of segmentation to do in the loss function:
    # 'n': None- the loss is applied to the entire image.
    # 'p': Predicted: The loss is applied where the prediction segments.
    # 'g': Ground truth: The loss is applied based on the GT segmentation.
    # 'o': The loss is based on GT == 1.0
    lsg='n',
    # [lsn]: What kind of loss to apply for the normals.
    lsn='a',
    # [pnf] The predict normal's frame. 'w' for world or 'c' for cam
    pnf='c',
    # [nlw]: The normal prediction loss weight.
    nlw=1.0,
    # [ni] Whether to add the normals as a secondary input.
    ni='f',
    # [nmo]: Whether to *only* input the normals:
    nmo='f',
    # [dxw]: locals xyz loss weight
    dxw=1.0,
    # [dnw]: locals normal loss weight
    dnw=1.0,
    # [drl]: The type of depth regression loss:
    drl='l2',
    # [msg]: What kind of segmentation to do in the depth -> XYZ mapping:
    # 'd': Do the mapping based on where the predicted depth is nontrivial.
    # 'p': Do the mapping based on the predicted segmentation mask.
    # 'g': Do the mapping based on the ground truth segmentation mask.
    msg='p',
    # [lpe]: Whether to have a local pointnet encoder. Only works if an xyz
    # image is available at test time:
    lpe='f',
    # [lpn]: Whether to add normals as a feature for the local pointnets.
    lpn='f',
    # [lpt]: The threshold (in radii) of the local pointclouds.
    lpt=4.0,
    # [vf]: Whether to add a point validity one-hot feature to the pointnet.
    vf='f',
    # [fod]: Whether to fix the object_detection encoder defaults.
    fod='t',
    # [ftd]: Whether to finetune the decoder
    ftd='f',
    # # [iz]: Whether to ignore 0s in the local pointnet encoders.
    # iz='f',
    lyr=0,
    # Whether to include global scope in the final feature vector.
    grf='f',
    # The task identifier, for looking up backwards compatible hparams (below):
    ident='rgb2q')


autoencoder_hparams = tf.contrib.training.HParams(
    # [ob]: Whether to optimize the blobs.
    ob='t',
    # [cp]: The constant prediction mode.
    # 'a': abs
    # 's': sigmoid.
    cp='a',
    # [dbg]: Whether to run in debug mode (checks for NaNs/Infs).
    dbg='f',
    # [bs]: The batch size.
    bs=24,  # 8,
    # [h]: The height of the input observation.
    h=137,
    # [w]: The width of the input observation.
    w=137,
    # [lr]: The learning rate.
    lr=5e-5,
    # Whether to 'nerfify' the occnet inputs:
    hyo='f',
    # Whether to 'nerfify' the pointnet inputs:
    hyp='f',
    # [loss]: A string encoding the loss to apply. Concatenate to apply several.
    #   'u': The uniform sample loss from the paper.
    #   'ns': The near surface sample loss from the paper.
    #   'ec': The element center loss from the paper.
    loss='unsbbgi',
    # [lrf]: The loss receptive field.
    #   'g': Global: the loss is based on the global reconstruction.
    #   'l': Local: The loss is based on the local reconstruction.
    #   'x': Local Points: The loss is local points locally reconstructed.
    lrf='g',
    # [arch]: A string encoding the model architecture to use:
    #  'efcnn': A simple early-fusion feed-forward CNN.
    #  'ttcnn': A two-tower early-fusion feed-forward CNN. One tower for
    #           explicit parameters, and one tower for implicit parameters.
    arch='efcnn',
    # [cnna]: The network architecture of the CNN, if applicable.
    #  'cnn': A simple cnn with 5 conv layers and 5 FC layers.
    #  'r18': ResNet18.
    #  'r50': ResNet50.
    #  'h50': Tf-hub ImageNet pretrained ResNet50
    cnna='r50',
    # [da]: The type of data augmentation to apply. Not compatible with
    # arch=efcnn, because depth maps can't be augmented.
    # 'f': No data augmentation.
    # 'p': Panning rotations.
    # 'r': Random SO(3) rotations.
    # 't': Random transformations with centering, rotations, and scaling.
    da='f',
    # [cri]: Whether to crop the region in the input
    cri='f',
    # [crl]: Whether to crop the region in the loss
    crl='f',
    # [cic]: The input crop count.
    cic=1024,
    # [clc]: The supervision crop count.
    clc=1024,
    # [ia]: The architecture for implicit parameter prediction.
    #  '1': Predict both implicit and explicit parameters with the same network.
    #  '2': Predict implicit and explicit parameters with two separate networks.
    #  'p': Predict implicit parameters with local pointnets.
    ia='p',
    # [p64]: Whether to enable the pointnet 64-d transformation.
    p64='f',
    # [fua]: Whether to enable 3D rotation of the model latents.
    fua='t',
    # [ipe]: Whether to enable implicit parameters. 't' or 'f'.
    ipe='f',
    # [ip]: The type of implicit parameters to use.
    #  'sb': Predict a minivector associated with each shape element that is
    #        residual to the prediction ('strictly better').
    ip='sb',
    # [ipc]: Whether the implicit parameters actually contribute (for debugging)
    # 't': Yes
    # 'f': No.
    ipc='t',
    # [ips]: The size of the implicit parameter vector.
    ips=32,
    # [npe]: Whether there is a separate network per shape element.
    #  't': Each shape element has its own neural network.
    #  'f': The shape elements share a single network.
    npe='f',
    # [pe]: The point encoder to use.
    # 'pn': PointNet
    # 'dg': DG-CNN
    pe='pn',
    # [nf]: The DG-CNN feature count.
    nf=64,
    # [fbp]: Whether to fix the very slow pointnet reduce.
    fbp='t',
    # [mfc]: The maxpool feature count for PointNet.
    mfc=1024,
    # [udp]: Whether to use the deprecated PointNet with Conv 2Ds.
    udp='t',
    # [orc]: The number of ResNet layers in OccNet
    orc=1,
    # [sc]: The number of shape elements.
    sc=100,
    # [lyr]: The number of blobs with left-right symmetry.
    lyr=10,
    # [opt]: The optimizer to use.
    #   'adm': Adam.
    #   'sgd': SGD.
    opt='adm',
    # [tx]: How to transform to element coordinate frames when generating
    # OccNet sample points.
    #   'crs': Center, rotate, scale. Any subset of these is fine, but they
    #     will be applied as SRC so 'cs' probably doesn't make much sense.
    #   'i': Identity (i.e. global coordinates).
    tx='crs',
    # [lhdn]: Whether to learn the sigmoid normalization parameter.
    lhdn='f',
    # [nrm]: The type of batch normalization to apply:
    #   'none': Do not apply any batch normalization.
    #   'bn': Enables batch norm, sets trainable to true, and sets is_training
    #     to true only for the train task.
    nrm='none',
    # [samp]: The input sampling scheme for the model:
    #   'imd': A dodecahedron of depth images.
    #   'imdpn': A dodecahedron of depth images and a point cloud with normals.
    #   'im1d': A single depth image from the dodecahedron.
    #   'rgb': A single rgb image from a random position.
    #   'imrd': One or more depth images from the dodecahedron.
    samp='imdpn',
    # [bg]: The background color for rgb images:
    # 'b': Black.
    # 'ws': White-smooth.
    bg='ws',
    # [lpc]: The number of local points per frame.
    lpc=1024,
    # [rc]: The random image count, if random image samples are used.
    rc=1,
    # [spc]: The sample point count. If a sparse loss sampling strategy is
    #   selected, then this is the number of random points to sample.
    spc=1024,
    # [xsc]: The lrf='x' pre-sample point count. The number of samples taken
    # from the larger set before again subsampling to [spc] samples.
    xsc=10000,
    # [sync]: 't' or 'f'. Whether to synchronize the GPUs during training, which
    #   increases the effective batch size and avoid stale gradients at the
    #   expense of decreased performance.
    sync='f',
    # [gpu]: If 'sync' is true, this should be set to the number of GPUs used in
    #   training; otherwise it is ignored.
    gpuc=0,
    # [vbs]: The virtual batch size; only used if 'sync' is true. The number of
    #   training examples to pool before applying a gradient.
    vbs=64,
    # [r]: 'iso' for isotropic, 'aa' for anisotropic and axis-aligned to the
    #   normalized mesh coordinates, 'cov' for general Gaussian RBFs.
    r='cov',
    # [ir]: When refining a prediction with gradient descent, which points to
    #   use. Still needs to be refactored with the rest of the eval code.
    ir='zero-set',
    # [res]: The rescaling between input and training SDFs.
    res=1.0,
    # [pou]: Whether the representation is a Partition of Unity. Either 't' or
    #   'f'. If 't', the sum is normalized by the sum of the weights. If 'f',
    #   it is not.
    pou='f',
    # [didx]: When only one of the dodecahedron images is shown to the network,
    #   the index of the image to show.
    didx=1,
    # [gh]: The height of the gaps (depth and luminance) images.
    gh=224,
    # [gw]: The width of the gaps (depth and luminance) images.
    gw=224,
    # [ucf]: The upweighting factor for interior points relative to exterior
    #   points.
    ucf=1.0,
    # [lset]: The isolevel of F defined to be the surface.
    lset=-0.07,
    # [igt]: The inside-the-grid threshold in the element center lowres grid
    #   inside loss.
    igt=0.044,  # Based on a 32^3 voxel grid with an extent of [-0.7, 0.7]
    # [wm]: Waymo scaling:
    wm='f',
    # [rsl]: The rescaling factor for the dataset.
    rsl=1.0,
    # [ig]: The weight on the element center lowres grid inside loss.
    ig=1.0,
    # [gs]: The weight on the element center lowres grid squared loss.
    gs=0.0,
    # [gd]: The weight on the element center lowres grid direct loss.
    gd=0.0,
    # [cm]: The weight on the loss that says that element centers should have
    #   a small l2 norm.
    cm=0.01,
    # [cc]: The weight on the component of the center loss that says that
    #   element centers must be inside the predicted surface.
    cc=0.0,
    # [vt]: The threshold for variance in the element center variance loss.
    vt=0.0,
    # [nnt]: The threshold for nearest neighbors in the element center nn loss.
    nnt=0.0,
    # [vw]: The weight on the center variance loss
    vw=0.0,
    # [nw]: The weight on the center nearest neighbor loss.
    nw=0.0,
    # [ow]: The weight on the overlap loss:
    ow=0.0,
    # [dd]: Whether to turn on the deprecated single-shape occnet decoder.
    #   't' or 'f'.
    dd='f',
    # [fon]: Whether to fix the occnet pre-cbn residual bug. 't' or 'f'.
    fon='t',
    # [itc]: The input pipeline thread count.
    itc=12,
    # [dmn]: The amount of noise in the depth map(s).
    dmn=0.0,
    # [pcn]: The amount of noise in the point cloud (along the normals).
    pcn=0.0,
    # [xin]: The amount of noise in the xyz image (in any direction).
    xin=0.0,
    # [hdn]: The 'hardness' of the soft classification transfer function. The
    #   larger the value, the more like a true 0/1 class label the transferred
    #   F values will be, but the shorter the distance until the gradient is
    #   within roundoff error.
    hdn=100.0,
    # [ag]: The blob aggregation method. 's' for sum, 'm' for max.
    ag='s',
    # [l2w]: The weight on the 'Uniform Sample Loss' from the paper.
    l2w=1.0,
    # [a2w]: The weight on the 'Near Surface Sample Loss' from the paper.
    a2w=0.1,
    # [fcc]: The number of fully connected layers after the first embedding
    #   vector, including the final linear layer.
    fcc=3,
    # [fcs]: The width of the fully connected layers that are immediately before
    #   the linear layer.
    fcs=2048,
    # [ibblw]: The weight on the component of the center loss that says that
    #   element centers must be inside the ground truth shape bounding box.
    ibblw=10.0,
    # [aucf]: Whether to apply the ucf hyperparameter to the near-surface sample
    #  loss. 't' or 'f'.
    aucf='f',
    # [cd]: Whether to cache the input data once read. 't' or 'f'.
    cd='f',
    # [elr]: The relative encoder learning rate (relative to the main LR)
    elr=1.0,
    # [cat]: The class to train on.
    cat='all',
    # A unique identifying string for this hparam dictionary
    ident='sif',
    # Whether to balance the categories. Requires a batch size multiplier of 13
    blc='t',
    # [rec]: The reconstruction equations
    rec='r',
    # The version of the implicit embedding CNN architecture.
    iec='v2')


def backwards_compatible_hparam_defaults(ident):
  """New hparams also need an entry here for existing models on disk."""
  mapping = {
      'sif': {
          'elr': 1.0,
          'cd': 'f',
          'dd': 'f',
          'fon': 't',
          'itc': 10,
          'ia': '2',
          'lpc': 1024,
          'p64': 't',
          'iec': 'v1',
          'lrf': 'g',
          'lyr': 0,
          'bg': 'b',
          'blc': 'f',
          'wm': 'f',
          'rsl': 1.0,
          'rec': 'r',
          'dbg': 'f',
          'cp': 'a',
          'tx': 'i',
          'dmn': 0.0,
          'pcn': 0.0,
          'xin': 0.0,
          'ag': 's',
          'vt': 0.0,
          'nnt': 0.0,
          'vw': 0.0,
          'nw': 0.0,
          'ow': 0.0,
          'pe': 'pn',
          'nf': 64,
          'fbp': 'f',
          'mfc': 1024,
          'udp': 't',
          'da': 'f',
          'cri': 'f',
          'cic': 1024,
          'crl': 'f',
          'clc': 1024,
          'hyo': 'f',
          'hyp': 'f',
      },
      'rgb2q': {
          'l': 'l2',
          'w': 'u',
          'sv': 'a',
          'bg': 'b',
          'ilw': 1.0,
          'dlw': 1.0,
          'wh': 'f',
          'rlw': 1.0,
          'slw': 0.0,
          'tlw': 0.0,
          'xlw': 0.0,
          'xt': '',
          'res': 137,
          'dec': 'dl',
          'et': 'f',
          'im': 'rgb',
          'fod': 'f',
          'eil': 'f',
          'spt': 't',
          'txd': 'f',
          'up': 't',
          'rdx': 'f',
          'st': 'f',
          'lsg': 'n',
          'msg': 'd',
          'sec': 'epr50',
          'drl': 'l2',
          'lpe': 'f',
          'ed': 10,
          'lpt': 4.0,
          'vf': 'f',
          'ftd': 'f',
          'lyr': 0,
          'lpn': 'f',
          'grf': 'f',
          'ni': 'f',
          'nmo': 'f',
          # 'iz':'f',
      }
  }
  return mapping[ident]


def tf_hparams_to_dict(tf_hparams):
  hparams = vars(tf_hparams)
  d = {}
  for k, v in hparams.items():
    if k == '_hparam_types' or k == '_model_structure':
      continue
    d[k] = v
  return d


def build_ldif_hparams():
  """Sets hyperparameters according to the published LDIF results."""
  d = tf_hparams_to_dict(autoencoder_hparams)
  d['arch'] = 'efcnn'
  d['samp'] = 'imdpn'
  d['cnna'] = 's50'
  d['hyo'] = 'f'
  d['hyp'] = 'f'
  d['sc'] = 32
  d['lyr'] = 16
  d['loss'] = 'unsbbgi'
  d['mfc'] = 512
  d['ips'] = 32
  d['ipe'] = 't'
  return tf.contrib.training.HParams(**d)


def build_sif_hparams():
  """The SIF representation trained according to the original SIF paper."""
  d = tf_hparams_to_dict(autoencoder_hparams)
  d['h'] = 137
  d['w'] = 137
  d['samp'] = 'im1dpn'
  d['ia'] = '1'
  d['cnna'] = 'cnn'
  d['sc'] = 100
  d['lyr'] = 0
  d['cat'] = 'all'
  d['cd'] = 'f'
  d['r'] = 'aa'
  d['spc'] = 3000
  d['blc'] = 'f'
  d['bs'] = 16
  d['ips'] = 0
  d['ipe'] = 'f'
  d['ucf'] = 10.0
  d['aucf'] = 't'
  d['loss'] = 'unsec'
  d['cm'] = 0.1
  d['igt'] = 0.0
  d['ig'] = 0.0
  d['gs'] = 0.0
  d['gd'] = 0.0
  d['cc'] = 0.01
  return tf.contrib.training.HParams(**d)


def build_improved_sif_hparams():
  """A better version of SIF using architecture improvements from LDIF."""
  d = tf_hparams_to_dict(autoencoder_hparams)
  d['arch'] = 'efcnn'
  d['samp'] = 'imdpn'
  d['cnna'] = 's50'
  d['hyo'] = 'f'
  d['hyp'] = 'f'
  d['sc'] = 32
  d['lyr'] = 16
  d['loss'] = 'unsbbgi'
  d['mfc'] = 512
  d['ips'] = 0
  d['ipe'] = 'f'
  return tf.contrib.training.HParams(**d)


def build_singleview_depth_hparams():
  """A singleview-depth architecture."""
  d = tf_hparams_to_dict(autoencoder_hparams)
  d['h'] = 224
  d['w'] = 224
  d['cnna'] = 's50'
  d['samp'] = 'im1xyzpn'
  d['didx'] = 0
  d['ia'] = 'p'
  d['sc'] = 32
  d['lyr'] = 16
  return tf.contrib.training.HParams(**d)


def write_hparams(hparams, path):
  d = tf_hparams_to_dict(hparams)
  file_util.writebin(path, pickle.dumps(d))
  return


def read_hparams_with_new_backwards_compatible_additions(path):
  """Reads hparams from a file and adds in any new backwards compatible ones."""
  kvs = pickle.loads(file_util.readbin(path))
  log.verbose('Loaded %s' % repr(kvs))
  if 'ident' not in kvs:
    ident = 'sif'  #  A default identifier for old checkpoints.
    log.info('Default ident!')
  else:
    ident = kvs['ident']
  new_additions = backwards_compatible_hparam_defaults(ident)
  for k, v in new_additions.items():
    if k not in kvs:
      log.verbose('Adding hparam %s:%s since it postdates the checkpoint.' %
                  (k, str(v)))
      kvs[k] = v
  # r50 is no longer supported, replace it:
  if 'cnna' in kvs and kvs['cnna'] == 'r50':
    kvs['cnna'] = 's50'
  hparams = tf.contrib.training.HParams(**kvs)

  return hparams


def read_hparams(path):
  kvs = pickle.loads(file_util.readbin(path))
  hparams = tf.contrib.training.HParams(**kvs)
  return hparams

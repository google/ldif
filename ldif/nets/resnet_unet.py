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
"""A ResNet-Unet with skip connections."""

import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training, data_format):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return inputs  # TODO(kgenova) Why is this early-return here?
  return tf.layers.batch_normalization(  # pylint: disable=unreachable
      inputs=inputs,
      axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON,
      center=True,
      scale=True,
      training=training,
      fused=True)


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or [batch,
      height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
      Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(
        inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


################################################################################
# ResNet block definitions.
################################################################################
def _building_block_v1(inputs, filters, training, projection_shortcut, strides,
                       data_format):
  """A single block for ResNet v1, without a bottleneck.

  Convolution then batch normalization then ReLU as described by:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or [batch,
      height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference mode.
      Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = batch_norm(
        inputs=shortcut, training=training, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs


def _building_block_v2(inputs, filters, training, projection_shortcut, strides,
                       data_format):
  """A single block for ResNet v2, without a bottleneck.

  Batch normalization then ReLu then convolution as described by:
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or [batch,
      height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference mode.
      Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)

  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format)

  return inputs + shortcut


def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
                         strides, data_format):
  """A single block for ResNet v1, with a bottleneck.

  Similar to _building_block_v1(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or [batch,
      height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference mode.
      Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = batch_norm(
        inputs=shortcut, training=training, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
                         strides, data_format):
  """A single block for ResNet v2, with a bottleneck.

  Similar to _building_block_v2(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
  Adapted to the ordering conventions of:
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or [batch,
      height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference mode.
      Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)

  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)

  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)

  return inputs + shortcut


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or [batch,
      height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the model.
      Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """

  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * 4 if bottleneck else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                    data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, training, None, 1, data_format)

  return tf.identity(inputs, name)


def resize(inputs, height, width, data_format):
  if data_format == 'channels_first':
    inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])
  else:
    assert data_format == 'channels_last'
  output_res = tf.stack([height, width])
  output = tf.image.resize_images(inputs, output_res, align_corners=True)
  if data_format == 'channels_first':
    output = tf.transpose(output, perm=[0, 3, 1, 2])
  return output


def assert_shape(tensor, shape, name):
  """Fails an assert if the tensor fails the shape compatibility check."""
  real_shape = tensor.get_shape().as_list()
  same_rank = len(real_shape) == len(shape)
  values_different = [
      1 for i in range(min(len(shape), len(real_shape)))
      if shape[i] != real_shape[i] and shape[i] != -1
  ]
  all_equal = not values_different
  if not same_rank or not all_equal:
    log.info(
        'Error: Expected tensor %s to have shape %s, but it had shape %s.' %
        (name, str(shape), str(real_shape)))
    assert False


def resnet_unet(inputs, scope, param_count, is_training, model_config):
  """The main interface to the resnet_unet architecture."""
  with tf.variable_scope(scope):
    assert len(inputs.get_shape().as_list()) == 4
    batch_size, input_height, input_width, feature_count = inputs.get_shape(
    ).as_list()
    del feature_count

    if input_height != 224 or input_width != 224:
      inputs = tf.image.resize_images(
          inputs,
          tf.stack([224, 224]),
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
          align_corners=True)
    tf.summary.image('input_imd0', inputs)
    # We probably want to get the input to 224x224 if we want to do the standard
    # resnet approach.
    resnet_size = 50
    bottleneck = resnet_size >= 50
    num_classes = 1024
    num_filters = 64
    kernel_size = 7  # Was 7 in imagenet
    conv_stride = 2
    first_pool_size = 3
    first_pool_stride = 2
    # blocks_per_set = (resnet_size - 2) // 6
    blockset_block_counts = [3, 4, 6, 3]
    resnet_version = 2
    # data_format = 'channels_first' if is_training else 'channels_last'
    data_format = 'channels_last'
    model = Model(
        resnet_size,
        bottleneck,
        num_classes,
        num_filters,
        kernel_size,
        conv_stride,
        first_pool_size,
        first_pool_stride,
        blockset_block_counts,
        block_strides=[1, 2, 2, 2],
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=tf.float32)

    if data_format == 'channels_first':
      fullres_in = tf.transpose(inputs, perm=[0, 3, 1, 2])
    else:
      fullres_in = inputs

    feature_axis = 1 if data_format == 'channels_first' else 3
    post_fc, intermediate_outputs = model(inputs, is_training)
    bs = batch_size
    assert_shape(post_fc, [bs, 1024], 'unet:post_fc')

    flat_predict = model_config.hparams.fp
    if flat_predict == 't':
      post_fc = tf.nn.relu(post_fc)
      prediction = tf.layers.dense(
          inputs=post_fc, units=param_count * model_config.hparams.sc)
      prediction = tf.reshape(
          prediction, [batch_size, model_config.hparams.sc, param_count])
      return prediction

    input_at_4x4 = resize(fullres_in, 4, 4, data_format)  # pylint: disable=unused-variable
    input_at_7x7 = resize(fullres_in, 7, 7, data_format)
    input_at_14x14 = resize(fullres_in, 14, 14, data_format)
    input_at_28x28 = resize(fullres_in, 28, 28, data_format)
    input_at_56x56 = resize(fullres_in, 56, 56, data_format)

    skip_from_7x7 = intermediate_outputs['post_block_3']
    skip_from_14x14 = intermediate_outputs['post_block_2']
    skip_from_28x28 = intermediate_outputs['post_block_1']
    skip_from_56x56 = intermediate_outputs['post_block_0']

    if data_format == 'channels_first':
      assert_shape(skip_from_7x7, [bs, 2048, 7, 7], 'unet:skip_from_7x7')
      assert_shape(skip_from_14x14, [bs, 1024, 14, 14], 'unet:skip_from_14x14')
      assert_shape(skip_from_28x28, [bs, 512, 28, 28], 'unet:skip_from_28x28')
      assert_shape(skip_from_56x56, [bs, 256, 56, 56], 'unet:skip_from_56x56')
    else:
      assert_shape(skip_from_7x7, [bs, 7, 7, 2048], 'unet:skip_from_7x7')
      assert_shape(skip_from_14x14, [bs, 14, 14, 1024], 'unet:skip_from_14x14')
      assert_shape(skip_from_28x28, [bs, 28, 28, 512], 'unet:skip_from_28x28')
      assert_shape(skip_from_56x56, [bs, 56, 56, 256], 'unet:skip_from_56x56')

    # The 1024 feature vector
    features_4x4 = tf.reshape(post_fc, [bs, 4, 4, 64])
    if data_format == 'channels_first':
      features_4x4 = tf.transpose(features_4x4, perm=[0, 3, 1, 2])
    features_4x4 = block_layer(
        inputs=features_4x4,
        filters=1024,
        bottleneck=bottleneck,
        block_fn=_bottleneck_block_v2,
        blocks=3,
        strides=1,
        training=is_training,
        name='up_block_layer_4x4',
        data_format=data_format)

    output_4x4 = conv2d_fixed_padding(
        inputs=features_4x4,
        filters=param_count,
        kernel_size=1,
        strides=1,
        data_format=data_format)
    if data_format == 'channels_first':
      assert_shape(output_4x4, [bs, param_count, 4, 4], 'unet:4x4 out')
    else:
      assert_shape(output_4x4, [bs, 4, 4, param_count], 'unet:4x4 out')

    # Move up to 7x7:
    output_4x4_at_7x7 = resize(output_4x4, 7, 7, data_format)
    features_4x4_at_7x7 = resize(features_4x4, 7, 7, data_format)
    features_7x7 = tf.concat(
        [output_4x4_at_7x7, input_at_7x7, features_4x4_at_7x7, skip_from_7x7],
        axis=feature_axis)
    features_7x7 = block_layer(
        inputs=features_7x7,
        filters=512,
        bottleneck=bottleneck,
        block_fn=_bottleneck_block_v2,
        blocks=3,
        strides=1,
        training=is_training,
        name='up_block_layer_7x7',
        data_format=data_format)
    output_7x7 = conv2d_fixed_padding(
        inputs=features_7x7,
        filters=param_count,
        kernel_size=1,
        strides=1,
        data_format=data_format)

    # Move up to 14x14:
    output_7x7_at_14x14 = resize(output_7x7, 14, 14, data_format)
    features_7x7_at_14x14 = resize(features_7x7, 14, 14, data_format)
    features_14x14 = tf.concat([
        output_7x7_at_14x14, input_at_14x14, features_7x7_at_14x14,
        skip_from_14x14
    ],
                               axis=feature_axis)
    features_14x14 = block_layer(
        inputs=features_14x14,
        filters=256,
        bottleneck=bottleneck,
        block_fn=_bottleneck_block_v2,
        blocks=3,
        strides=1,
        training=is_training,
        name='up_block_layer_14x14',
        data_format=data_format)
    output_14x14 = conv2d_fixed_padding(
        inputs=features_14x14,
        filters=param_count,
        kernel_size=1,
        strides=1,
        data_format=data_format)

    # Move up to 28x28
    output_14x14_at_28x28 = resize(output_14x14, 28, 28, data_format)
    features_14x14_at_28x28 = resize(features_14x14, 28, 28, data_format)
    features_28x28 = tf.concat([
        output_14x14_at_28x28, input_at_28x28, features_14x14_at_28x28,
        skip_from_28x28
    ],
                               axis=feature_axis)
    features_28x28 = block_layer(
        inputs=features_28x28,
        filters=128,
        bottleneck=bottleneck,
        block_fn=_bottleneck_block_v2,
        blocks=3,
        strides=1,
        training=is_training,
        name='up_block_layer_28x28',
        data_format=data_format)
    output_28x28 = conv2d_fixed_padding(
        inputs=features_28x28,
        filters=param_count,
        kernel_size=1,
        strides=1,
        data_format=data_format)

    # Move up to 56x56 (finally):
    output_28x28_at_56x56 = resize(output_28x28, 56, 56, data_format)
    features_28x28_at_56x56 = resize(features_28x28, 56, 56, data_format)
    features_56x56 = tf.concat([
        output_28x28_at_56x56, input_at_56x56, features_28x28_at_56x56,
        skip_from_56x56
    ],
                               axis=feature_axis)
    features_56x56 = block_layer(
        inputs=features_56x56,
        filters=128,
        bottleneck=bottleneck,
        block_fn=_bottleneck_block_v2,
        blocks=3,
        strides=1,
        training=is_training,
        name='up_block_layer_56x56',
        data_format=data_format)
    output_56x56 = conv2d_fixed_padding(
        inputs=features_56x56,
        filters=param_count,
        kernel_size=1,
        strides=1,
        data_format=data_format)

    if data_format == 'channels_first':
      flat_out_4x4 = tf.reshape(
          tf.transpose(output_4x4, perm=[0, 2, 3, 1]), [bs, 4 * 4, param_count])
      flat_out_7x7 = tf.reshape(
          tf.transpose(output_7x7, perm=[0, 2, 3, 1]), [bs, 7 * 7, param_count])
      flat_out_14x14 = tf.reshape(
          tf.transpose(output_14x14, perm=[0, 2, 3, 1]),
          [bs, 14 * 14, param_count])
      flat_out_28x28 = tf.reshape(
          tf.transpose(output_28x28, perm=[0, 2, 3, 1]),
          [bs, 28 * 28, param_count])
      flat_out_56x56 = tf.reshape(
          tf.transpose(output_56x56, perm=[0, 2, 3, 1]),
          [bs, 56 * 56, param_count])
    else:
      flat_out_4x4 = tf.reshape(output_4x4, [bs, 4 * 4, param_count])
      flat_out_7x7 = tf.reshape(output_7x7, [bs, 7 * 7, param_count])
      flat_out_14x14 = tf.reshape(output_14x14, [bs, 14 * 14, param_count])
      flat_out_28x28 = tf.reshape(output_28x28, [bs, 28 * 28, param_count])
      flat_out_56x56 = tf.reshape(output_56x56, [bs, 56 * 56, param_count])

    return [
        flat_out_4x4, flat_out_7x7, flat_out_14x14, flat_out_28x28,
        flat_out_56x56
    ]


class Model(object):
  """Base class for building the Resnet Model."""

  def __init__(self,
               resnet_size,
               bottleneck,
               num_classes,
               num_filters,
               kernel_size,
               conv_stride,
               first_pool_size,
               first_pool_stride,
               block_sizes,
               block_strides,
               resnet_version=DEFAULT_VERSION,
               data_format=None,
               dtype=DEFAULT_DTYPE):
    """Creates a model for classifying an image.

    Args:
      resnet_size: A single integer for the size of the ResNet model.
      bottleneck: Use regular blocks or bottleneck blocks.
      num_classes: The number of classes used as labels.
      num_filters: The number of filters to use for the first block layer of the
        model. This number is then doubled for each subsequent block layer.
      kernel_size: The kernel size to use for convolution.
      conv_stride: stride size for the initial convolutional layer
      first_pool_size: Pool size to be used for the first pooling layer. If
        none, the first pooling layer is skipped.
      first_pool_stride: stride size for the first pooling layer. Not used if
        first_pool_size is None.
      block_sizes: A list containing n values, where n is the number of sets of
        block layers desired. Each value should be the number of blocks in the
        i-th set.
      block_strides: List of integers representing the desired stride size for
        each of the sets of block layers. Should be same length as block_sizes.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      data_format: Input format ('channels_last', 'channels_first', or None). If
        set to None, the format is dependent on whether a GPU is available.
      dtype: The TensorFlow dtype to use for calculations. If not specified
        tf.float32 is used.

    Raises:
      ValueError: if invalid version is selected.
    """
    self.resnet_size = resnet_size

    if not data_format:
      data_format = ('channels_first'
                     if tf.test.is_built_with_cuda() else 'channels_last')

    self.resnet_version = resnet_version
    if resnet_version not in (1, 2):
      raise ValueError(
          'Resnet version should be 1 or 2. See README for citations.')

    self.bottleneck = bottleneck
    if bottleneck:
      if resnet_version == 1:
        self.block_fn = _bottleneck_block_v1
      else:
        self.block_fn = _bottleneck_block_v2
    else:
      if resnet_version == 1:
        self.block_fn = _building_block_v1
      else:
        self.block_fn = _building_block_v2

    if dtype not in ALLOWED_TYPES:
      raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

    self.data_format = data_format
    self.num_classes = num_classes
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.conv_stride = conv_stride
    self.first_pool_size = first_pool_size
    self.first_pool_stride = first_pool_stride
    self.block_sizes = block_sizes
    self.block_strides = block_strides
    self.dtype = dtype
    self.pre_activation = resnet_version == 2

  def _custom_dtype_getter(self,
                           getter,
                           name,
                           shape=None,
                           dtype=DEFAULT_DTYPE,
                           *args,
                           **kwargs):  # pylint: disable=keyword-arg-before-vararg
    """Creates variables in fp32, then casts to fp16 if necessary.

    This function is a custom getter. A custom getter is a function with the
    same signature as tf.get_variable, except it has an additional getter
    parameter. Custom getters can be passed as the `custom_getter` parameter of
    tf.variable_scope. Then, tf.get_variable will call the custom getter,
    instead of directly getting a variable itself. This can be used to change
    the types of variables that are retrieved with tf.get_variable.
    The `getter` parameter is the underlying variable getter, that would have
    been called if no custom getter was used. Custom getters typically get a
    variable with `getter`, then modify it in some way.
    This custom getter will create an fp32 variable. If a low precision
    (e.g. float16) variable was requested it will then cast the variable to the
    requested dtype. The reason we do not directly create variables in low
    precision dtypes is that applying small gradients to such variables may
    cause the variable not to change.
    Args:
      getter: The underlying variable getter, that has the same signature as
        tf.get_variable and returns a variable.
      name: The name of the variable to get.
      shape: The shape of the variable to get.
      dtype: The dtype of the variable to get. Note that if this is a low
        precision dtype, the variable will be created as a tf.float32 variable,
        then cast to the appropriate dtype
      *args: Additional arguments to pass unmodified to getter.
      **kwargs: Additional keyword arguments to pass unmodified to getter.

    Returns:
      A variable which is cast to fp16 if necessary.
    """

    if dtype in CASTABLE_TYPES:
      var = getter(name, shape, tf.float32, *args, **kwargs)
      return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
      return getter(name, shape, dtype, *args, **kwargs)

  def _model_variable_scope(self):
    """Returns a variable scope that the model should be created under.

    If self.dtype is a castable type, model variable will be created in fp32
    then cast to self.dtype before being used.
    Returns:
      A variable scope for the model.
    """

    return tf.variable_scope(
        'resnet_model', custom_getter=self._custom_dtype_getter)

  def __call__(self, inputs, training):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """
    intermediate_outputs = {}

    with self._model_variable_scope():
      if self.data_format == 'channels_first':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

      inputs = conv2d_fixed_padding(
          inputs=inputs,
          filters=self.num_filters,
          kernel_size=self.kernel_size,
          strides=self.conv_stride,
          data_format=self.data_format)
      inputs = tf.identity(inputs, 'initial_conv')
      intermediate_outputs['post_initial_conv'] = inputs

      # We do not include batch normalization or activation functions in V2
      # for the initial conv1 because the first ResNet unit will perform these
      # for both the shortcut and non-shortcut paths as part of the first
      # block's projection. Cf. Appendix of [2].
      if self.resnet_version == 1:
        inputs = batch_norm(inputs, training, self.data_format)
        inputs = tf.nn.relu(inputs)

      if self.first_pool_size:
        inputs = tf.layers.max_pooling2d(
            inputs=inputs,
            pool_size=self.first_pool_size,
            strides=self.first_pool_stride,
            padding='SAME',
            data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')
        intermediate_outputs['post_initial_max_pool'] = inputs

      for i, num_blocks in enumerate(self.block_sizes):
        num_filters = self.num_filters * (2**i)
        inputs = block_layer(
            inputs=inputs,
            filters=num_filters,
            bottleneck=self.bottleneck,
            block_fn=self.block_fn,
            blocks=num_blocks,
            strides=self.block_strides[i],
            training=training,
            name='block_layer{}'.format(i + 1),
            data_format=self.data_format)
        intermediate_outputs['post_block_%i' % i] = inputs

      # Only apply the BN and ReLU for model that does pre_activation in each
      # building/bottleneck block, eg resnet V2.
      if self.pre_activation:
        inputs = batch_norm(inputs, training, self.data_format)
        inputs = tf.nn.relu(inputs)

      # The current top layer has shape
      # `batch_size x pool_size x pool_size x final_size`.
      # ResNet does an Average Pooling layer over pool_size,
      # but that is the same as doing a reduce_mean. We do a reduce_mean
      # here because it performs better than AveragePooling2D.
      axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
      inputs = tf.reduce_mean(inputs, axes, keepdims=True)
      inputs = tf.identity(inputs, 'final_reduce_mean')

      inputs = tf.squeeze(inputs, axes)
      intermediate_outputs['post_avgpool'] = inputs
      inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
      inputs = tf.identity(inputs, 'final_dense')
      intermediate_outputs['post_fc'] = inputs
      return inputs, intermediate_outputs

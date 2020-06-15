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
"""Utilities for mathematical operations."""

import itertools
import math

import numpy as np

import tensorflow as tf


def int_log2(i):
  """Computes the floor of the base 2 logarithm of an integer."""
  log2 = 0
  while i >= 2:
    log2 += 1
    i = i >> 1
  return log2


def nonzero_mean(tensor):
  """The mean over nonzero values in a tensor."""
  num = tf.reduce_sum(tensor)
  denom = tf.cast(tf.count_nonzero(tensor), dtype=tf.float32)
  denom = tf.where(denom == 0.0, 1e-8, denom)
  return tf.divide(num, denom)


def increase_frequency(t, out_dim, flatten=False, interleave=True):
  """Maps elements of a tensor to a higher frequency, higher dimensional space.

  As shown in NeRF (https://arxiv.org/pdf/2003.08934.pdf), this can help
  networks learn higher frequency functions more easily since they are typically
  biased to low frequency functions. By increasing the frequency of the input
  signal, such biases are mitigated.

  Args:
    t: Tensor with any shape. Type tf.float32. The normalization of the input
      dictates how many dimensions are needed to avoid periodicity. The NeRF
      paper normalizes all inputs to the range [0, 1], which is safe.
    out_dim: How many (sine, cosine) pairs to generate for each element of t.
      Referred to as 'L' in NeRF. Integer.
    flatten: Whether to flatten the output tensor to have the same rank as t.
      Boolean. See returns section for details.
    interleave: Whether to interleave the sin and cos results, as described in
      the paper. If true, then the vector will contain [sin(2^0*t_i*pi),
      cos(2^0*t_i*pi), sin(2^1*t_i*pi), ...]. If false, some operations will be
      avoided, but the order will be [sin(2^0*t_i*pi), sin(2^1*t_i*pi), ...,
      cos(2^0*t_i*pi), cos(2^1*t_i*pi), ...].

  Returns:
    Tensor of type tf.float32. Has shape [..., out_dim*2] if flatten is false.
    If flatten is true, then if t has shape [..., N] then the output will have
    shape [..., N*out_dim*2].
  """
  # TODO(kgenova) Without a custom kernel this is somewhat less efficient,
  # because the sin and cos results have to be next to one another in the output
  # but tensorflow only allows computing them with two different ops. Thus it is
  # necessary to do some expensive tf.concats. It probably won't be a bottleneck
  # in most pipelines.
  t = math.pi * t
  scales = np.power(2, np.arange(out_dim, dtype=np.int32)).astype(np.float32)
  t_rank = len(t.shape)
  scale_shape = [1] * t_rank + [out_dim]
  scales = tf.constant(np.reshape(scales, scale_shape), dtype=tf.float32)
  scaled = tf.expand_dims(t, axis=-1) * scales
  sin = tf.sin(scaled)
  cos = tf.cos(scaled)
  output = tf.concat([sin, cos], axis=-1)
  if interleave:
    sines = tf.unstack(sin, axis=-1)
    cosines = tf.unstack(cos, axis=-1)
    output = tf.stack(list(itertools.chain(*zip(sines, cosines))), axis=-1)
  if flatten:
    t_shape = t.get_shape().as_list()
    output = tf.reshape(output, t_shape[:-1] + [t_shape[-1] * out_dim * 2])
  return output

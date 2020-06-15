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
"""Utilties for iterative optimization in tensorflow."""

import tensorflow as tf


def clip_by_global_norm_custom(t_list, clip_norm, use_norm=None, name=None):
  """Clips values of multiple tensors by the ratio of the sum of their norms.

  Given a tuple or list of tensors `t_list`, and a clipping ratio `clip_norm`,
  this operation returns a list of clipped tensors `list_clipped`
  and the global norm (`global_norm`) of all tensors in `t_list`. Optionally,
  if you've already computed the global norm for `t_list`, you can specify
  the global norm with `use_norm`.
  To perform the clipping, the values `t_list[i]` are set to:
      t_list[i] * clip_norm / max(global_norm, clip_norm)
  where:
      global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
  If `clip_norm > global_norm` then the entries in `t_list` remain as they are,
  otherwise they're all shrunk by the global ratio.
  Any of the entries of `t_list` that are of type `None` are ignored.
  This is the correct way to perform gradient clipping (for example, see
  [Pascanu et al., 2012](http://arxiv.org/abs/1211.5063)
  ([pdf](http://arxiv.org/pdf/1211.5063.pdf))).
  However, it is slower than `clip_by_norm()` because all the parameters must be
  ready before the clipping operation can be performed.
  Args:
    t_list: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
    clip_norm: A 0-D (scalar) `Tensor` > 0. The clipping ratio.
    use_norm: A 0-D (scalar) `Tensor` of type `float` (optional). The global
      norm to use. If not provided, `global_norm()` is used to compute the norm.
    name: A name for the operation (optional).

  Returns:
    list_clipped: A list of `Tensors` of the same type as `list_t`.
    global_norm: A 0-D (scalar) `Tensor` representing the global norm.
  Raises:
    TypeError: If `t_list` is not a sequence.
    InvalidArgumentError: If global norm is not finite.
  """
  # if (not isinstance(t_list, collections.Sequence)
  #     or isinstance(t_list, six.string_types)):
  #   raise TypeError("t_list should be a sequence")
  t_list = list(t_list)
  if use_norm is None:
    use_norm = tf.linalg.global_norm(t_list, name)

  with tf.name_scope(name, 'clip_by_global_norm_custom',
                     t_list + [clip_norm]) as name:
    # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
    scale = clip_norm * tf.minimum(
        1.0 / use_norm,
        tf.constant(1.0, dtype=use_norm.dtype) / clip_norm)

    values = [
        tf.convert_to_tensor(
            t.values if isinstance(t, tf.IndexedSlices) else t, name='t_%d' % i)
        if t is not None else t for i, t in enumerate(t_list)
    ]

    values_clipped = []
    for i, v in enumerate(values):
      if v is None:
        values_clipped.append(None)
      else:
        values_clipped.append(tf.identity(v * scale, name='%s_%d' % (name, i)))

    list_clipped = [
        tf.IndexedSlices(c_v, t.indices, t.dense_shape) if isinstance(
            t, tf.IndexedSlices) else c_v
        for (c_v, t) in zip(values_clipped, t_list)
    ]

  return list_clipped, use_norm


def clip_by_global_norm(grads_and_vars):
  grads, variables = list(zip(*grads_and_vars))
  clipped_grads, _ = clip_by_global_norm_custom(grads, 5.0)
  return list(zip(clipped_grads, variables))

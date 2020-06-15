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
"""Code for evaluating models as they train."""

import time

import numpy as np
import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.model import hparams
from ldif.training import metrics
from ldif.inference import refine
from ldif.util import file_util
from ldif.util import image_util
from ldif.util import np_util
# pylint: enable=g-bad-import-order


def make_eval_step(model_config, training_example, prediction):
  """Returns a function that computes a model evaluation step."""
  # TODO(kgenova) The eval code hasn't really been refactored yet, only train.
  opt_mode = model_config.hparams.ir
  if opt_mode == 'zero-set':
    # TODO(kgenova) This is broken:
    sample_locations = training_example.sample_sdf_near_surface(3000)[0]
    point_count = sample_locations.get_shape().as_list()[1]
    optimization_target = tf.zeros([model_config.hparams.bs, point_count, 1],
                                   dtype=tf.float32)
  else:
    raise ValueError('Unrecognized value for hparam ir: %s' %
                     model_config.hparams.ir)

  with tf.name_scope('eval_summaries'):
    # Summarize the outputs:
    if len(prediction.structured_implicit.tensor_list) == 3:
      constants, centers, radii = prediction.structured_implicit.tensor_list
      iparams = None
    elif len(prediction.structured_implicit.tensor_list) == 4:
      constants, centers, radii, iparams = (
          prediction.structured_implicit.tensor_list)
    mean_radius = tf.reduce_mean(radii)
    tf.summary.scalar('%s-mean_radius' % training_example.split, mean_radius)
    mean_center = tf.reduce_mean(centers)
    tf.summary.scalar('%s-mean_center' % training_example.split, mean_center)
    mean_constant = tf.reduce_mean(constants)
    tf.summary.scalar('%s-mean_constant' % training_example.split,
                      mean_constant)
    if iparams is not None:
      tf.summary.scalar('%s-mean_abs_iparams' % training_example.split,
                        tf.reduce_mean(tf.abs(iparams)))
      tf.summary.histogram('%s-iparams' % training_example.split, iparams)
    tf.summary.histogram('%s-constants' % training_example.split, constants)
    tf.summary.histogram('%s-centers' % training_example.split, centers)
    tf.summary.histogram('%s-radii' % training_example.split, radii)

  (structured_implicit_ph, samples_ph, optimization_target_ph, gradients,
   render_summary_op, original_vis_ph,
   iterated_render_tf) = refine.sample_gradients_wrt_placeholders(
       model_config, training_example, prediction, sample_locations)

  vis_count = 1
  do_iterative_update = False

  big_render_ph = tf.placeholder(
      tf.uint8)
  # [model_config.hparams.bs * vis_count, 256 * 1, 256 + 256 * 2, 4])
  big_render_summary_op = tf.summary.image(
      '%s-big-render' % training_example.split,
      big_render_ph,
      collections=[],
      max_outputs=model_config.hparams.bs * vis_count)

  rbf_render_at_half_ph = tf.placeholder(tf.float32)
  rbf_render_at_half_summary_op = tf.summary.image(
      '%s-rbf_render_at_half' % training_example.split,
      rbf_render_at_half_ph,
      collections=[],
      max_outputs=model_config.hparams.bs * vis_count)

  depth_gt_out_ph = tf.placeholder(tf.float32)
  depth_gt_out_summary_op = tf.summary.image(
      '%s-depth_gt_out' % training_example.split,
      depth_gt_out_ph,
      collections=[],
      max_outputs=model_config.hparams.bs * vis_count)

  # Also prefetches in case this is a property mutating the graph:
  in_out_image_big = tf.image.resize_images(
      prediction.in_out_image,
      size=[256, 256],
      align_corners=True)
  tf.logging.info('in_out_image_big shape: %s',
                  str(in_out_image_big.get_shape().as_list()))

  sample_locations, sample_gt = training_example.all_uniform_samples()
  example_iou = metrics.point_iou(prediction.structured_implicit,
                                  sample_locations, sample_gt, model_config)
  iou_ph = tf.placeholder(tf.float32)
  mean_iou_summary_op = tf.summary.scalar(
      '%s-mean-iou' % training_example.split,
      tf.reduce_mean(iou_ph),
      collections=[])

  iou_hist_summary_op = tf.summary.histogram(
      '%s-iou-histogram' % training_example.split, iou_ph, collections=[])

  def eval_step(session, global_step, desired_num_examples, eval_tag,
                eval_checkpoint):
    """Runs a single eval step.

    Runs over the full desired eval set, regardless of batch size.

    Args:
      session: A tf.Session instance.
      global_step: The global step tensor.
      desired_num_examples: The number of examples from the eval dataset to
        evaluate.
      eval_tag: A tag to specify the eval type. Defaults to 'eval'.
      eval_checkpoint: The path of the checkpoint being evaluated.

    Returns:
      A list of tf.Summary objects computed during the eval.
    """
    step_start_time = time.time()
    del eval_tag, desired_num_examples
    global_step_int = int(global_step)
    # num_batches = max(1, desired_num_examples // model_config.hparams.bs)
    big_render_images = []
    all_centers_np = []
    all_radii_np = []
    all_constants_np = []
    all_quadrics_np = []
    all_iparams_np = []
    all_mesh_names_np = []
    all_depth_images_np = []
    tf.logging.info('The eval checkpoint str is %s', eval_checkpoint)

    eval_dir = '/'.join(eval_checkpoint.split('/')[:-1])

    hparam_path = eval_dir + '/hparam_pickle.txt'
    if not file_util.exists(hparam_path):
      hparams.write_hparams(model_config.hparams, hparam_path)
    output_dir = (eval_dir + '/eval-step-' + str(global_step_int) + '/')

    def to_uint8(np_im):
      return (np.clip(255.0 * np_im, 0, 255.0)).astype(np.uint8)

    ran_count = 0
    max_run_count = 500
    ious = np.zeros(max_run_count, dtype=np.float32)
    # Run until the end of the dataset:
    for vi in range(max_run_count):
      tf.logging.info('Starting eval item %i, total elapsed time is %0.2f...',
                      vi,
                      time.time() - step_start_time)
      try:
        vis_start_time = time.time()
        if vi < vis_count:
          misc_tensors_to_eval = [
              model_config.summary_op, optimization_target, sample_locations,
              in_out_image_big, training_example.mesh_name, example_iou,
          ]
          np_out = session.run(misc_tensors_to_eval +
                               prediction.structured_implicit.tensor_list)
          (summaries, optimization_target_np, samples_np, in_out_image_big_np,
           mesh_names_np, example_iou_np) = np_out[:len(misc_tensors_to_eval)]
          in_out_image_big_np = np.reshape(in_out_image_big_np,
                                           [256, 256, 1])
          in_out_image_big_np = image_util.get_pil_formatted_image(
              in_out_image_big_np)
          tf.logging.info('in_out_image_big_np shape: %s',
                          str(in_out_image_big_np.shape))
          in_out_image_big_np = np.reshape(in_out_image_big_np,
                                           [1, 256, 256, 4])
          implicit_np_list = np_out[len(misc_tensors_to_eval):]
          tf.logging.info('\tElapsed after first sess run: %0.2f',
                          time.time() - vis_start_time)
        else:
          np_out = session.run([training_example.mesh_name, example_iou] +
                               prediction.structured_implicit.tensor_list)
          mesh_names_np = np_out[0]
          example_iou_np = np_out[1]
          implicit_np_list = np_out[2:]

        # TODO(kgenova) It would be nice to move all this functionality into
        # a numpy StructuredImplicitNp class, and hide these internals.

        ious[ran_count] = example_iou_np
        ran_count += 1

        constants_np, centers_np, radii_np = implicit_np_list[:3]
        if len(implicit_np_list) == 4:
          iparams_np = implicit_np_list[3]
        else:
          iparams_np = None
        # For now, just map to quadrics and move on:
        quadrics_np = np.zeros(
            [constants_np.shape[0], constants_np.shape[1], 4, 4])
        quadrics_np[0, :, 3, 3] = np.reshape(constants_np[0, :], [
            model_config.hparams.sc,
        ])

        all_centers_np.append(np.copy(centers_np))
        all_radii_np.append(np.copy(radii_np))
        all_constants_np.append(np.copy(constants_np))
        all_quadrics_np.append(np.copy(quadrics_np))
        all_mesh_names_np.append(mesh_names_np)
        if iparams_np is not None:
          all_iparams_np.append(iparams_np)

        # For most of the dataset, just do inference to get the representation.
        # Everything afterwards is just for tensorboard.
        if vi >= vis_count:
          continue

        visualize_with_marching_cubes = False
        if visualize_with_marching_cubes:
          # TODO(kgenova) This code is quite wrong now. If we want to enable it
          # it should be rewritten to call a structured_implicit_function to
          # handle evaluation (for instance the lset subtraction is bad).
          marching_cubes_ims_np, output_volumes = np_util.visualize_prediction(
              quadrics_np,
              centers_np,
              radii_np,
              renormalize=model_config.hparams.pou == 't',
              thresh=model_config.hparams.lset)
          tf.logging.info('\tElapsed after first visualize_prediction: %0.2f',
                          time.time() - vis_start_time)
          offset_marching_cubes_ims_np, _ = np_util.visualize_prediction(
              quadrics_np,
              centers_np,
              radii_np,
              renormalize=model_config.hparams.pou == 't',
              thresh=0.1,
              input_volumes=output_volumes)
          tf.logging.info('\tElapsed after second visualize_prediction: %0.2f',
                          time.time() - vis_start_time)
          tf.logging.info('About to concatenate shapes: %s, %s, %s',
                          str(in_out_image_big_np.shape),
                          str(marching_cubes_ims_np.shape),
                          str(offset_marching_cubes_ims_np.shape))
          in_out_image_big_np = np.concatenate([
              in_out_image_big_np, marching_cubes_ims_np,
              offset_marching_cubes_ims_np
          ],
                                               axis=2)

        if do_iterative_update:
          # This code will fail (it's left unasserted to give a helpful tf error
          # message). The tensor it creates will now be the wrong size.
          render_summary, iterated_render_np = refine.refine(
              structured_implicit_ph, optimization_target_ph, samples_ph,
              original_vis_ph, gradients, implicit_np_list,
              optimization_target_np, samples_np, in_out_image_big_np, session,
              render_summary_op, iterated_render_tf)
          render_summary = [render_summary]
          in_out_with_iterated = np.concatenate(
              [in_out_image_big_np, iterated_render_np], axis=2)
          big_render_images.append(to_uint8(in_out_with_iterated))
        else:
          big_render_images.append(to_uint8(in_out_image_big_np))

          # TODO(kgenova) Is this really the depth image?
          depth_image_np = in_out_image_big_np[:, :, :256, :]
          all_depth_images_np.append(depth_image_np)

          render_summary = []
      except tf.errors.OutOfRangeError:
        break
    tf.logging.info('Elapsed after vis loop: %0.2f',
                    time.time() - step_start_time)
    ious = ious[:ran_count]
    mean_iou_summary, iou_hist_summary = session.run(
        [mean_iou_summary_op, iou_hist_summary_op], feed_dict={iou_ph: ious})

    all_centers_np = np.concatenate(all_centers_np)
    all_radii_np = np.concatenate(all_radii_np)
    all_constants_np = np.concatenate(all_constants_np)
    all_quadrics_np = np.concatenate(all_quadrics_np)
    all_mesh_names_np = np.concatenate(all_mesh_names_np)
    all_depth_images_np = np.concatenate(all_depth_images_np)
    if all_iparams_np:
      all_iparams_np = np.concatenate(all_iparams_np)

    file_util.mkdir(output_dir, exist_ok=True)
    file_util.write_np(
        '%s/%s-constants.npy' % (output_dir, training_example.split),
        all_constants_np)
    file_util.write_np(
        '%s/%s-quadrics.npy' % (output_dir, training_example.split),
        all_quadrics_np)
    file_util.write_np(
        '%s/%s-centers.npy' % (output_dir, training_example.split),
        all_centers_np)
    file_util.write_np('%s/%s-radii.npy' % (output_dir, training_example.split),
                       all_radii_np)
    file_util.write_np(
        '%s/%s-mesh_names.npy' % (output_dir, training_example.split),
        all_mesh_names_np)
    # We do an explicit comparison because the type of all_iparams_np might
    # not be a list at this point:
    # pylint: disable=g-explicit-bool-comparison
    if all_iparams_np != []:
      file_util.write_np(
          '%s/%s-iparams.npy' % (output_dir, training_example.split),
          all_iparams_np)

    # Now that the full set predictions have been saved to disk, scrap
    # everything after the first vis_count:
    all_centers_np = all_centers_np[:vis_count, ...]
    all_radii_np = all_radii_np[:vis_count, ...]
    all_constants_np = all_constants_np[:vis_count, ...]
    all_mesh_names_np = all_mesh_names_np[:vis_count, ...]

    tf.logging.info('Elapsed after .npy save: %0.2f',
                    time.time() - step_start_time)

    rbf_renders_at_half = np_util.plot_rbfs_at_thresh(
        all_centers_np, all_radii_np, thresh=0.5)
    rbf_renders_at_half_summary = session.run(
        rbf_render_at_half_summary_op,
        feed_dict={rbf_render_at_half_ph: rbf_renders_at_half})
    tf.logging.info('Elapsed after rbf_at_half summary: %0.2f',
                    time.time() - step_start_time)
    tf.logging.info('All depth images shape: %s',
                    str(all_depth_images_np.shape))
    depth_gt_out_summary = session.run(
        depth_gt_out_summary_op,
        feed_dict={
            depth_gt_out_ph:
                np.concatenate([all_depth_images_np, all_depth_images_np],
                               axis=2)
        })
    tf.logging.info('Elapsed after depth_gt_out summary: %0.2f',
                    time.time() - step_start_time)

    big_render_summary = session.run(
        big_render_summary_op,
        feed_dict={big_render_ph: np.concatenate(big_render_images, axis=0)})
    tf.logging.info('Evaluated %d batches of size %d.', vis_count,
                    model_config.hparams.bs)
    tf.logging.info('Elapsed at end of step: %0.2f',
                    time.time() - step_start_time)
    return [
        summaries, big_render_summary, rbf_renders_at_half_summary,
        depth_gt_out_summary, mean_iou_summary, iou_hist_summary
    ] + render_summary

  return eval_step

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
"""Computes metrics given predicted and ground truth shape."""

import numpy as np
import pandas as pd
import scipy
import tabulate

# ldif is an internal package, and should be imported last.
# pylint: disable=g-bad-import-order
from ldif.inference import example
from ldif.util import file_util
from ldif.util import mesh_util
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order

OCCNET_FSCORE_EPS = 1e-09


def get_class(row):
  """Attempts to give synsets a nice name."""
  synset_to_cat = {
      # '02691156': 'airplane',
      '02933112': 'cabinet',
      '03001627': 'chair',
      '03636649': 'lamp',
      '04090263': 'rifle',
      '04379243': 'table',
      '04530566': 'watercraft',
      '02828884': 'bench',
      '02958343': 'car',
      '03211117': 'display',
      '03691459': 'speaker',
      '04256520': 'sofa',
      '04401088': 'telephone',
      '02472293': 'bodyshapes',  # Z-filled only. Normally there's no leading 0.
      '04330267': 'stove',
      '04004475': 'printer',
      '03928116': 'piano',
      '02942699': 'camera',
      '02843684': 'birdhouse',
      '02818832': 'bed',
      '04460130': 'tower',
      '03337140': 'file',
      '02871439': 'bookshelf',
      '03710193': 'mailbox',
  }
  key = row['key']
  if '|' in key:
    if len([x for x in key if x == '|']) != 1:
      raise ValueError(
          f"Couldn't parse {key} into class+mesh name using | delimiter.")
    maybe_synset1, maybe_synset2 = key.split('|')
  else:
    maybe_synset1, maybe_synset2 = row['key'].split('/')[:2]
  if maybe_synset1 in synset_to_cat:
    return synset_to_cat[maybe_synset1]
  if maybe_synset2 in synset_to_cat:
    return synset_to_cat[maybe_synset2]
  # The first value should be the synset, the second only exists for backwards
  # compatibility:
  return maybe_synset1


def print_pivot_table(class_mean_df, metric_name, metric_pretty_print):
  cmean = pd.pivot_table(
      class_mean_df, values=metric_name, index=['class'], columns=['xid'])
  log.info('%s:\n%s' %
           (metric_pretty_print,
            tabulate.tabulate(
                cmean, headers='keys', tablefmt='fancy_grid', floatfmt='.3f')))


def aggregate_extracted(csv_path_or_df):
  """Creates a summary of metrics from the full summary csv."""
  # This uses print, not log, so it is meant for post-beam pipeline use.
  if isinstance(csv_path_or_df, str):
    df = file_util.read_csv(csv_path_or_df)
  else:
    df = csv_path_or_df
  # The XID index is the leading number in the hparam string:
  df['class'] = df.apply(get_class, axis=1)

  class_means = df.groupby(['class']).mean().reset_index()

  mean_of_means = class_means.mean(axis=0)
  mean_of_means['class'] = 'mean'

  final = class_means.append(mean_of_means, ignore_index=True)
  log.info('\n' + tabulate.tabulate(
      final, headers='keys', tablefmt='fancy_grid', floatfmt='.3f'))
  log.info('\n' + tabulate.tabulate(
      final, headers='keys', tablefmt='latex', floatfmt='.3f'))
  return final


def sample_points_and_face_normals(mesh, sample_count):
  points, indices = mesh.sample(sample_count, return_index=True)
  points = points.astype(np.float32)
  normals = mesh.face_normals[indices]
  return points, normals


def pointcloud_neighbor_distances_indices(source_points, target_points):
  target_kdtree = scipy.spatial.cKDTree(target_points)
  distances, indices = target_kdtree.query(source_points, n_jobs=-1)
  return distances, indices


def dot_product(a, b):
  if len(a.shape) != 2:
    raise ValueError('Dot Product with input shape: %s' % repr(a.shape))
  if len(b.shape) != 2:
    raise ValueError('Dot Product with input shape: %s' % repr(b.shape))
  return np.sum(a * b, axis=1)


def point_iou(pred_is_inside, gt_is_inside):
  intersection = np.logical_and(pred_is_inside, gt_is_inside).astype(np.float32)
  union = np.logical_or(pred_is_inside, gt_is_inside).astype(np.float32)
  iou = 100.0 * np.sum(intersection) / (np.sum(union) + 1e-05)
  return iou


def point_metrics(element):
  """Adds all point-set metrics to the input dictionary."""
  if 'iou_predictions' not in element:
    raise ValueError('IoU requested by iou samples were not computed.')
  gt_is_inside = None
  example_np = element_to_example(element)
  pred_is_inside = element['iou_predictions'] < 0.5  # -0.07
  gt_is_inside = np.reshape(example_np.uniform_samples[..., -1],
                            [100000, 1]) < 0.0
  element['iou'] = point_iou(pred_is_inside, gt_is_inside)
  return element


def element_to_example(element):
  if 'rgb_path' in element:
    return example.InferenceExample.from_rgb_path_and_split(
        element['rgb_path'], element['split'])
  elif 'npz_path' in element:
    return example.InferencExample.from_npz_path(element['npz_path'])
  else:
    raise ValueError("Can't parse element: %s" % repr(element))


def percent_below(dists, thresh):
  return np.mean((dists**2 <= thresh).astype(np.float32)) * 100.0


def f_score(a_to_b, b_to_a, thresh):
  precision = percent_below(a_to_b, thresh)
  recall = percent_below(b_to_a, thresh)

  return (2 * precision * recall) / (precision + recall + OCCNET_FSCORE_EPS)


def fscore(mesh1,
           mesh2,
           sample_count=100000,
           tau=1e-04,
           points1=None,
           points2=None):
  """Computes the F-Score at tau between two meshes."""
  points1, points2 = get_points(mesh1, mesh2, points1, points2, sample_count)
  dist12, _ = pointcloud_neighbor_distances_indices(points1, points2)
  dist21, _ = pointcloud_neighbor_distances_indices(points2, points1)
  f_score_tau = f_score(dist12, dist21, tau)
  return f_score_tau


def mesh_chamfer_via_points(mesh1,
                            mesh2,
                            sample_count=100000,
                            points1=None,
                            points2=None):
  points1, points2 = get_points(mesh1, mesh2, points1, points2, sample_count)
  dist12, _ = pointcloud_neighbor_distances_indices(points1, points2)
  dist21, _ = pointcloud_neighbor_distances_indices(points2, points1)
  chamfer = 1000.0 * (np.mean(dist12**2) + np.mean(dist21**2))
  return chamfer


def get_points(mesh1, mesh2, points1, points2, sample_count):
  if points1 is not None or points2 is not None:
    assert points1 is not None and points2 is not None
  else:
    points1, _ = sample_points_and_face_normals(mesh1, sample_count)
    points2, _ = sample_points_and_face_normals(mesh2, sample_count)
  return points1, points2


def normal_consistency(mesh1, mesh2, sample_count=100000, return_points=False):
  """Computes the normal consistency metric between two meshes."""
  points1, normals1 = sample_points_and_face_normals(mesh1, sample_count)
  points2, normals2 = sample_points_and_face_normals(mesh2, sample_count)

  _, indices12 = pointcloud_neighbor_distances_indices(points1, points2)
  _, indices21 = pointcloud_neighbor_distances_indices(points2, points1)

  normals12 = normals2[indices12]
  normals21 = normals1[indices21]

  # We take abs because the OccNet code takes abs...
  nc12 = np.abs(dot_product(normals1, normals12))
  nc21 = np.abs(dot_product(normals2, normals21))
  nc = 0.5 * np.mean(nc12) + 0.5 * np.mean(nc21)
  if return_points:
    return nc, points1, points2
  return nc


def print_mesh_metrics(pred_mesh, gt_mesh, sample_count=100000):
  nc, fst, fs2t, chamfer = all_mesh_metrics(pred_mesh, gt_mesh, sample_count)
  log.info('F-Score (tau)   : %0.2f' % fst)
  log.info('F-Score (2*tau) : %0.2f' % fs2t)
  log.info('Normal Const.   : %0.2f' % nc)
  log.info('Chamfer Distance: %0.5f' % chamfer)


def compute_all(sif_vector, decoder, e, resolution=256, sample_count=100000):
  """Computes iou, f-score, f-score (2*tau), normal consistency, and chamfer."""
  iou = decoder.iou(sif_vector, e)
  pred_mesh, had_crossing = decoder.extract_mesh(
      sif_vector, resolution=resolution, return_success=True)
  if had_crossing:
    pred_mesh_occnet_frame = pred_mesh.apply_transform(e.gaps_to_occnet)
  else:
    # We just have a sphere, don't try to un-normalize it:
    pred_mesh_occnet_frame = pred_mesh
  gt_mesh_occnet_frame = e.gt_mesh
  nc, fst, fs2t, chamfer = all_mesh_metrics(pred_mesh_occnet_frame,
                                            gt_mesh_occnet_frame, sample_count)
  return {
      'iou': iou,
      'f_score_tau': fst,
      'f_score_2tau': fs2t,
      'chamfer': chamfer,
      'normal_c': nc
  }


def print_all(sif_vector, decoder, e, resolution=256, sample_count=100000):
  results = compute_all(sif_vector, decoder, e, resolution, sample_count)
  metrics = ''
  metrics += 'IoU             : %0.2f\n' % results['iou']
  metrics += 'F-Score (tau)   : %0.2f\n' % results['f_score_tau']
  metrics += 'F-Score (2*tau) : %0.2f\n' % results['f_score_2tau']
  metrics += 'Normal Const.   : %0.2f\n' % results['normal_c']
  metrics += 'Chamfer Distance: %0.5f\n' % results['chamfer']
  log.info(metrics)
  print(metrics)


def all_mesh_metrics(mesh1, mesh2, sample_count=100000):
  nc, points1, points2 = normal_consistency(
      mesh1, mesh2, sample_count, return_points=True)
  fs_tau = fscore(mesh1, mesh2, sample_count, 1e-04, points1, points2)
  fs_2tau = fscore(mesh1, mesh2, sample_count, 2.0 * 1e-04, points1, points2)
  chamfer = mesh_chamfer_via_points(mesh1, mesh2, sample_count, points1,
                                    points2)
  return nc, fs_tau, fs_2tau, chamfer


def mesh_metrics(element):
  """Computes the chamfer distance and normal consistency metrics."""
  log.info('Metric step input: %s' % repr(element))
  example_np = element_to_example(element)
  if not element['mesh_str']:
    raise ValueError(
        'Empty mesh string encountered for %s but mesh metrics required.' %
        repr(element))
  mesh = mesh_util.deserialize(element['mesh_str'])
  if mesh.is_empty:
    raise ValueError(
        'Empty mesh encountered for %s but mesh metrics required.' %
        repr(element))

  sample_count = 100000
  points_pred, normals_pred = sample_points_and_face_normals(mesh, sample_count)
  points_gt, normals_gt = sample_points_and_face_normals(
      example_np.gt_mesh, sample_count)

  pred_to_gt_dist, pred_to_gt_indices = pointcloud_neighbor_distances_indices(
      points_pred, points_gt)
  gt_to_pred_dist, gt_to_pred_indices = pointcloud_neighbor_distances_indices(
      points_gt, points_pred)

  pred_to_gt_normals = normals_gt[pred_to_gt_indices]
  gt_to_pred_normals = normals_pred[gt_to_pred_indices]

  # We take abs because the OccNet code takes abs
  pred_to_gt_normal_consistency = np.abs(
      dot_product(normals_pred, pred_to_gt_normals))
  gt_to_pred_normal_consistency = np.abs(
      dot_product(normals_gt, gt_to_pred_normals))

  # The 100 factor is because papers multiply by 100 for display purposes.
  chamfer = 100.0 * (np.mean(pred_to_gt_dist**2) + np.mean(gt_to_pred_dist**2))

  nc = 0.5 * np.mean(pred_to_gt_normal_consistency) + 0.5 * np.mean(
      gt_to_pred_normal_consistency)

  tau = 1e-04
  f_score_tau = f_score(pred_to_gt_dist, gt_to_pred_dist, tau)
  f_score_2tau = f_score(pred_to_gt_dist, gt_to_pred_dist, 2.0 * tau)

  element['chamfer'] = chamfer
  element['normal_consistency'] = nc
  element['f_score_tau'] = f_score_tau
  element['f_score_2tau'] = f_score_2tau
  element['split'] = example_np.split
  element['synset'] = example_np.synset
  element['name'] = example_np.mesh_hash
  element['class'] = example_np.cat
  return element

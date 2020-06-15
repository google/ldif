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
"""Tests for model.py."""

import os
import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import image_util
from ldif.util import line_util
from ldif.util import path_util
# pylint: enable=g-bad-import-order


class ModelTest(tf.test.TestCase):

  def setUp(self):
    super(ModelTest, self).setUp()
    self.test_data_directory = os.path.join(path_util.get_path_to_ldif_root(),
                                            'test_data')

  def test_render_centered_square(self):
    line_parameters = tf.constant([0.0, 64.0, 64.0, 32.0, 32.0],
                                  dtype=tf.float32)
    image = line_util.line_to_image(
        line_parameters, height=128, width=128, falloff=None)
    target_image_name = 'Centered_Square_0.png'
    baseline_image_path = os.path.join(self.test_data_directory,
                                       target_image_name)
    with self.test_session() as sess:
      image = image_util.get_pil_formatted_image(sess.run(image))
      image_util.expect_image_file_and_image_are_near(
          self, baseline_image_path, image, target_image_name,
          '%s does not match.' % target_image_name)

  def test_render_vertical_rectangle(self):
    line_parameters = tf.constant([0.0, 64.0, 64.0, 16.0, 48.0],
                                  dtype=tf.float32)
    image = line_util.line_to_image(
        line_parameters, height=128, width=128, falloff=None)
    target_image_name = 'Centered_Vertical_Rectangle_0.png'
    baseline_image_path = os.path.join(self.test_data_directory,
                                       target_image_name)
    with self.test_session() as sess:
      image = image_util.get_pil_formatted_image(sess.run(image))
      image_util.expect_image_file_and_image_are_near(
          self, baseline_image_path, image, target_image_name,
          '%s does not match.' % target_image_name)

  def test_render_offset_vertical_rectangle(self):
    line_parameters = tf.constant([0.0, 80.0, 49.0, 16.0, 48.0],
                                  dtype=tf.float32)
    image = line_util.line_to_image(
        line_parameters, height=128, width=128, falloff=None)
    target_image_name = 'Offset_Vertical_Rectangle_0.png'
    baseline_image_path = os.path.join(self.test_data_directory,
                                       target_image_name)
    with self.test_session() as sess:
      image = image_util.get_pil_formatted_image(sess.run(image))
      image_util.expect_image_file_and_image_are_near(
          self, baseline_image_path, image, target_image_name,
          '%s does not match.' % target_image_name)

  def test_render_offset_vertical_rectangle_rectangular_image(self):
    line_parameters = tf.constant([0.0, 80.0, 49.0, 16.0, 48.0],
                                  dtype=tf.float32)
    image = line_util.line_to_image(
        line_parameters, height=130, width=120, falloff=None)
    target_image_name = 'Offset_Vertical_Rectangle_1.png'
    baseline_image_path = os.path.join(self.test_data_directory,
                                       target_image_name)
    with self.test_session() as sess:
      image = image_util.get_pil_formatted_image(sess.run(image))
      image_util.expect_image_file_and_image_are_near(
          self, baseline_image_path, image, target_image_name,
          '%s does not match.' % target_image_name)

  def test_render_rotated_rectangle(self):
    line_parameters = tf.constant([3.14159 / 4.0, 64.0, 64.0, 16.0, 48.0],
                                  dtype=tf.float32)
    image = line_util.line_to_image(
        line_parameters, height=128, width=128, falloff=None)
    target_image_name = 'Rotated_Rectangle_0.png'
    baseline_image_path = os.path.join(self.test_data_directory,
                                       target_image_name)
    with self.test_session() as sess:
      image = image_util.get_pil_formatted_image(sess.run(image))
      image_util.expect_image_file_and_image_are_near(
          self, baseline_image_path, image, target_image_name,
          '%s does not match.' % target_image_name)

  def test_render_centered_square_with_blur(self):
    line_parameters = tf.constant([0.0, 64.0, 64.0, 16.0, 16.0],
                                  dtype=tf.float32)
    image = line_util.line_to_image(
        line_parameters, height=128, width=128, falloff=10.0)
    target_image_name = 'Centered_Square_Blur_0.png'
    baseline_image_path = os.path.join(self.test_data_directory,
                                       target_image_name)
    with self.test_session() as sess:
      image = image_util.get_pil_formatted_image(sess.run(image))
      image_util.expect_image_file_and_image_are_near(
          self, baseline_image_path, image, target_image_name,
          '%s does not match.' % target_image_name)

  def test_render_rotated_rectangle_with_blur(self):
    line_parameters = tf.constant([3.14159 / 4.0, 64.0, 64.0, 16.0, 48.0],
                                  dtype=tf.float32)
    image = line_util.line_to_image(
        line_parameters, height=128, width=128, falloff=10.0)
    target_image_name = 'Rotated_Rectangle_Blur_0.png'
    baseline_image_path = os.path.join(self.test_data_directory,
                                       target_image_name)
    with self.test_session() as sess:
      image = image_util.get_pil_formatted_image(sess.run(image))
      image_util.expect_image_file_and_image_are_near(
          self, baseline_image_path, image, target_image_name,
          '%s does not match.' % target_image_name)


if __name__ == '__main__':
  tf.test.main()

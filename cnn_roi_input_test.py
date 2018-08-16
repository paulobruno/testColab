# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for cifar10 input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import cnn_roi_input


class CIFAR10InputTest(tf.test.TestCase):

  def _record(self, label, gray):
    image_size = 16 * 16
    record = bytes(bytearray([label] + [gray] * image_size))
    expected = [[[gray]] * 16] * 16
    return record, expected

  def testSimple(self):
    labels = [1, 0, 0]
    records = [self._record(labels[0], 128),
               self._record(labels[1], 255),
               self._record(labels[2], 254)]
    contents = b"".join([record for record, _ in records])
    expected = [expected for _, expected in records]
    filename = os.path.join(self.get_temp_dir(), "cnn_roi")
    open(filename, "wb").write(contents)

    with self.test_session() as sess:
      q = tf.FIFOQueue(99, [tf.string], shapes=())
      q.enqueue([filename]).run()
      q.close().run()
      
      result = cnn_roi_input.read_cnn_roi(q)

      for i in range(3):
        key, label, uint8image = sess.run([
            result.key, result.label, result.uint8image])
        self.assertEqual("%s:%d" % (filename, i), tf.compat.as_text(key))
        self.assertEqual(labels[i], label)
        self.assertAllEqual(expected[i], uint8image)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run([result.key, result.uint8image])


if __name__ == "__main__":
  tf.test.main()

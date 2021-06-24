# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pytest

from dnc import util

# set seeds for determinism
np.random.seed(42)


class BatchInvertPermutation(tf.test.TestCase):
    def test(self):
        # Tests that the _batch_invert_permutation function correctly inverts a
        # batch of permutations.
        batch_size = 5
        length = 7

        permutations = np.empty([batch_size, length], dtype=int)
        for i in range(batch_size):
            permutations[i] = np.random.permutation(length)

        inverse = util.batch_invert_permutation(tf.constant(permutations, tf.int32))
        inverse = inverse.numpy()

        for i in range(batch_size):
            for j in range(length):
                self.assertEqual(permutations[i][inverse[i][j]], j)


class BatchGather(tf.test.TestCase):
    def test(self):
        values = np.array([[3, 1, 4, 1], [5, 9, 2, 6], [5, 3, 5, 7]])
        indexs = np.array([[1, 2, 0, 3], [3, 0, 1, 2], [0, 2, 1, 3]])
        target = np.array([[1, 4, 3, 1], [6, 5, 9, 2], [5, 5, 3, 7]])
        result = util.batch_gather(tf.constant(values), tf.constant(indexs))
        result = result.numpy()
        self.assertAllEqual(target, result)


@pytest.mark.parametrize(
    "batch_size, state_size, initial_state",
    [
        (2, [], []),
        (2, 2, tf.zeros([2, 2], dtype=tf.float32)),
        (
            2,
            [tf.TensorShape([1, 3]), 2],
            [tf.zeros([2, 1, 3], dtype=tf.float32), tf.zeros([2, 2], dtype=tf.float32)],
        ),
        (
            2,
            [2, [2, [tf.TensorShape([1, 3])]]],
            [
                tf.zeros([2, 2], dtype=tf.float32),
                [
                    tf.zeros([2, 2], dtype=tf.float32),
                    [tf.zeros([2, 1, 3], dtype=tf.float32)],
                ],
            ],
        ),
    ],
)
def test_initial_state_from_state_size(batch_size, state_size, initial_state):
    assert str(initial_state) == str(
        util.initial_state_from_state_size(state_size, batch_size, tf.float32)
    )

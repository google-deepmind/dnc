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
"""Tests for memory access."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn

import access
import util

BATCH_SIZE = 2
MEMORY_SIZE = 20
WORD_SIZE = 6
NUM_READS = 2
NUM_WRITES = 3
TIME_STEPS = 4
INPUT_SIZE = 10


class MemoryAccessTest(tf.test.TestCase):

  def setUp(self):
    self.module = access.MemoryAccess(MEMORY_SIZE, WORD_SIZE, NUM_READS,
                                      NUM_WRITES)
    self.initial_state = self.module.initial_state(BATCH_SIZE)

  def testBuildAndTrain(self):
    inputs = tf.random_normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])

    output, _ = rnn.dynamic_rnn(
        cell=self.module,
        inputs=inputs,
        initial_state=self.initial_state,
        time_major=True)

    targets = np.random.rand(TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE)
    loss = tf.reduce_mean(tf.square(output - targets))
    train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)
    init = tf.global_variables_initializer()

    with self.test_session():
      init.run()
      train_op.run()

  def testValidReadMode(self):
    inputs = self.module._read_inputs(
        tf.random_normal([BATCH_SIZE, INPUT_SIZE]))
    init = tf.global_variables_initializer()

    with self.test_session() as sess:
      init.run()
      inputs = sess.run(inputs)

    # Check that the read modes for each read head constitute a probability
    # distribution.
    self.assertAllClose(inputs['read_mode'].sum(2),
                        np.ones([BATCH_SIZE, NUM_READS]))
    self.assertGreaterEqual(inputs['read_mode'].min(), 0)

  def testWriteWeights(self):
    memory = 10 * (np.random.rand(BATCH_SIZE, MEMORY_SIZE, WORD_SIZE) - 0.5)
    usage = np.random.rand(BATCH_SIZE, MEMORY_SIZE)

    allocation_gate = np.random.rand(BATCH_SIZE, NUM_WRITES)
    write_gate = np.random.rand(BATCH_SIZE, NUM_WRITES)
    write_content_keys = np.random.rand(BATCH_SIZE, NUM_WRITES, WORD_SIZE)
    write_content_strengths = np.random.rand(BATCH_SIZE, NUM_WRITES)

    # Check that turning on allocation gate fully brings the write gate to
    # the allocation weighting (which we will control by controlling the usage).
    usage[:, 3] = 0
    allocation_gate[:, 0] = 1
    write_gate[:, 0] = 1

    inputs = {
        'allocation_gate': tf.constant(allocation_gate),
        'write_gate': tf.constant(write_gate),
        'write_content_keys': tf.constant(write_content_keys),
        'write_content_strengths': tf.constant(write_content_strengths)
    }

    weights = self.module._write_weights(inputs,
                                         tf.constant(memory),
                                         tf.constant(usage))

    with self.test_session():
      weights = weights.eval()

    # Check the weights sum to their target gating.
    self.assertAllClose(np.sum(weights, axis=2), write_gate, atol=5e-2)

    # Check that we fully allocated to the third row.
    weights_0_0_target = util.one_hot(MEMORY_SIZE, 3)
    self.assertAllClose(weights[0, 0], weights_0_0_target, atol=1e-3)

  def testReadWeights(self):
    memory = 10 * (np.random.rand(BATCH_SIZE, MEMORY_SIZE, WORD_SIZE) - 0.5)
    prev_read_weights = np.random.rand(BATCH_SIZE, NUM_READS, MEMORY_SIZE)
    prev_read_weights /= prev_read_weights.sum(2, keepdims=True) + 1

    link = np.random.rand(BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE)
    # Row and column sums should be at most 1:
    link /= np.maximum(link.sum(2, keepdims=True), 1)
    link /= np.maximum(link.sum(3, keepdims=True), 1)

    # We query the memory on the third location in memory, and select a large
    # strength on the query. Then we select a content-based read-mode.
    read_content_keys = np.random.rand(BATCH_SIZE, NUM_READS, WORD_SIZE)
    read_content_keys[0, 0] = memory[0, 3]
    read_content_strengths = tf.constant(
        100., shape=[BATCH_SIZE, NUM_READS], dtype=tf.float64)
    read_mode = np.random.rand(BATCH_SIZE, NUM_READS, 1 + 2 * NUM_WRITES)
    read_mode[0, 0, :] = util.one_hot(1 + 2 * NUM_WRITES, 2 * NUM_WRITES)
    inputs = {
        'read_content_keys': tf.constant(read_content_keys),
        'read_content_strengths': read_content_strengths,
        'read_mode': tf.constant(read_mode),
    }
    read_weights = self.module._read_weights(inputs, memory, prev_read_weights,
                                             link)
    with self.test_session():
      read_weights = read_weights.eval()

    # read_weights for batch 0, read head 0 should be memory location 3
    self.assertAllClose(
        read_weights[0, 0, :], util.one_hot(MEMORY_SIZE, 3), atol=1e-3)

  def testGradients(self):
    inputs = tf.constant(np.random.randn(BATCH_SIZE, INPUT_SIZE), tf.float32)
    output, _ = self.module(inputs, self.initial_state)
    loss = tf.reduce_sum(output)

    tensors_to_check = [
        inputs, self.initial_state.memory, self.initial_state.read_weights,
        self.initial_state.linkage.precedence_weights,
        self.initial_state.linkage.link
    ]
    shapes = [x.get_shape().as_list() for x in tensors_to_check]
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      err = tf.test.compute_gradient_error(tensors_to_check, shapes, loss, [1])
      self.assertLess(err, 0.1)


if __name__ == '__main__':
  tf.test.main()

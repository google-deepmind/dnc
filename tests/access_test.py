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

from dnc import access, addressing, util

BATCH_SIZE = 2
MEMORY_SIZE = 20
WORD_SIZE = 6
NUM_READS = 2
NUM_WRITES = 3
TIME_STEPS = 4
INPUT_SIZE = 10

DTYPE=tf.float32

# set seeds for determinism
np.random.seed(42)
from tensorflow.python.framework import random_seed
random_seed.set_seed(42)

class MemoryAccessTest(tf.test.TestCase):

  def setUp(self):
    self.cell = access.MemoryAccess(
        MEMORY_SIZE, WORD_SIZE, NUM_READS, NUM_WRITES)
    
    self.module = tf.keras.layers.RNN(
        cell=self.cell,
        time_major=True)

  def testBuildAndTrain(self):
    inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE], dtype=DTYPE)
    targets = np.random.rand(TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE)
    loss = lambda outputs, targets: tf.reduce_mean(input_tensor=tf.square(outputs - targets))
    print(self.module.get_initial_state(inputs))
    import ipdb; ipdb.set_trace()
    with tf.GradientTape() as tape:
        outputs, _ = self.module(
            inputs=inputs,
            #initial_state=self.initial_state,
        )
        loss_value = loss(outputs, targets)
        gradients = tape.gradient(loss_value, self.module.trainable_variables)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    optimizer.apply_gradients(zip(gradients, self.module.trainable_variables))

  def testValidReadMode(self):
    inputs = self.cell._read_inputs(
        tf.random.normal([BATCH_SIZE, INPUT_SIZE], dtype=DTYPE))

    # Check that the read modes for each read head constitute a probability
    # distribution.
    self.assertAllClose(inputs['read_mode'].numpy().sum(2),
                        np.ones([BATCH_SIZE, NUM_READS]))
    self.assertGreaterEqual(inputs['read_mode'].numpy().min(), 0)

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
        'allocation_gate': tf.constant(allocation_gate, dtype=DTYPE),
        'write_gate': tf.constant(write_gate, dtype=DTYPE),
        'write_content_keys': tf.constant(write_content_keys, dtype=DTYPE),
        'write_content_strengths': tf.constant(write_content_strengths, dtype=DTYPE)
    }

    weights = self.cell._write_weights(inputs,
                                         tf.constant(memory, dtype=DTYPE),
                                         tf.constant(usage, dtype=DTYPE))

    weights = weights.numpy()

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
        100., shape=[BATCH_SIZE, NUM_READS], dtype=DTYPE)
    read_mode = np.random.rand(BATCH_SIZE, NUM_READS, 1 + 2 * NUM_WRITES)
    read_mode[0, 0, :] = util.one_hot(1 + 2 * NUM_WRITES, 2 * NUM_WRITES)
    inputs = {
        'read_content_keys': tf.constant(read_content_keys, dtype=DTYPE),
        'read_content_strengths': read_content_strengths,
        'read_mode': tf.constant(read_mode, dtype=DTYPE),
    }
    read_weights = self.cell._read_weights(
        inputs,
        tf.cast(memory, dtype=DTYPE),
        tf.cast(prev_read_weights, dtype=DTYPE),
        tf.cast(link, dtype=DTYPE),
    )
    read_weights = read_weights.numpy()


    # read_weights for batch 0, read head 0 should be memory location 3
    self.assertAllClose(
        read_weights[0, 0, :], util.one_hot(MEMORY_SIZE, 3), atol=1e-3)

  def testGradients(self):
    inputs = tf.constant(np.random.randn(BATCH_SIZE, INPUT_SIZE), dtype=DTYPE)
    initial_state = self.module.get_initial_state(inputs)

    def evaluate_module(inputs, memory, read_weights, precedence_weights, link):
        initial_state = access.AccessState(
            memory=memory,
            read_weights=read_weights,
            write_weights=initial_state[access.WRITE_WEIGHTS],
            linkage=addressing.TemporalLinkageState(
                precedence_weights=precedence_weights,
                link=link
            ),
            usage=initial_state[access.USAGE],
        )
        output, _ = self.module(inputs, initial_state)
        loss = tf.reduce_sum(input_tensor=output)
        return loss

    tensors_to_check = [
        inputs,
        initial_state[access.MEMORY],
        initial_state[access.READ_WEIGHTS],
        initial_state[access.LINKAGE][addressing.PRECEDENCE_WEIGHTS],
        initial_state[access.LINKAGE][addressing.LINK],
    ]

    theoretical, numerical = tf.test.compute_gradient(
        evaluate_module,
        tensors_to_check,
        delta=1e-5
    )
    self.assertLess(
        sum([tf.norm(numerical[i] - theoretical[i]) for i in range(2)]),
        0.02,
        tensors_to_check
    )

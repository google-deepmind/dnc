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
"""Tests for memory addressing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sonnet as snt
import tensorflow as tf

import addressing
import util


class WeightedSoftmaxTest(tf.test.TestCase):

  def testValues(self):
    batch_size = 5
    num_heads = 3
    memory_size = 7

    activations_data = np.random.randn(batch_size, num_heads, memory_size)
    weights_data = np.ones((batch_size, num_heads))

    activations = tf.placeholder(tf.float32,
                                 [batch_size, num_heads, memory_size])
    weights = tf.placeholder(tf.float32, [batch_size, num_heads])
    # Run weighted softmax with identity placed on weights. Output should be
    # equal to a standalone softmax.
    observed = addressing.weighted_softmax(activations, weights, tf.identity)
    expected = snt.BatchApply(
        module_or_op=tf.nn.softmax, name='BatchSoftmax')(activations)
    with self.test_session() as sess:
      observed = sess.run(
          observed,
          feed_dict={activations: activations_data,
                     weights: weights_data})
      expected = sess.run(expected, feed_dict={activations: activations_data})
      self.assertAllClose(observed, expected)


class CosineWeightsTest(tf.test.TestCase):

  def testShape(self):
    batch_size = 5
    num_heads = 3
    memory_size = 7
    word_size = 2

    module = addressing.CosineWeights(num_heads, word_size)
    mem = tf.placeholder(tf.float32, [batch_size, memory_size, word_size])
    keys = tf.placeholder(tf.float32, [batch_size, num_heads, word_size])
    strengths = tf.placeholder(tf.float32, [batch_size, num_heads])
    weights = module(mem, keys, strengths)
    self.assertTrue(weights.get_shape().is_compatible_with(
        [batch_size, num_heads, memory_size]))

  def testValues(self):
    batch_size = 5
    num_heads = 4
    memory_size = 10
    word_size = 2

    mem_data = np.random.randn(batch_size, memory_size, word_size)
    np.copyto(mem_data[0, 0], [1, 2])
    np.copyto(mem_data[0, 1], [3, 4])
    np.copyto(mem_data[0, 2], [5, 6])

    keys_data = np.random.randn(batch_size, num_heads, word_size)
    np.copyto(keys_data[0, 0], [5, 6])
    np.copyto(keys_data[0, 1], [1, 2])
    np.copyto(keys_data[0, 2], [5, 6])
    np.copyto(keys_data[0, 3], [3, 4])
    strengths_data = np.random.randn(batch_size, num_heads)

    module = addressing.CosineWeights(num_heads, word_size)
    mem = tf.placeholder(tf.float32, [batch_size, memory_size, word_size])
    keys = tf.placeholder(tf.float32, [batch_size, num_heads, word_size])
    strengths = tf.placeholder(tf.float32, [batch_size, num_heads])
    weights = module(mem, keys, strengths)

    with self.test_session() as sess:
      result = sess.run(
          weights,
          feed_dict={mem: mem_data,
                     keys: keys_data,
                     strengths: strengths_data})

      # Manually checks results.
      strengths_softplus = np.log(1 + np.exp(strengths_data))
      similarity = np.zeros((memory_size))

      for b in xrange(batch_size):
        for h in xrange(num_heads):
          key = keys_data[b, h]
          key_norm = np.linalg.norm(key)

          for m in xrange(memory_size):
            row = mem_data[b, m]
            similarity[m] = np.dot(key, row) / (key_norm * np.linalg.norm(row))

          similarity = np.exp(similarity * strengths_softplus[b, h])
          similarity /= similarity.sum()
          self.assertAllClose(result[b, h], similarity, atol=1e-4, rtol=1e-4)

  def testDivideByZero(self):
    batch_size = 5
    num_heads = 4
    memory_size = 10
    word_size = 2

    module = addressing.CosineWeights(num_heads, word_size)
    keys = tf.random_normal([batch_size, num_heads, word_size])
    strengths = tf.random_normal([batch_size, num_heads])

    # First row of memory is non-zero to concentrate attention on this location.
    # Remaining rows are all zero.
    first_row_ones = tf.ones([batch_size, 1, word_size], dtype=tf.float32)
    remaining_zeros = tf.zeros(
        [batch_size, memory_size - 1, word_size], dtype=tf.float32)
    mem = tf.concat((first_row_ones, remaining_zeros), 1)

    output = module(mem, keys, strengths)
    gradients = tf.gradients(output, [mem, keys, strengths])

    with self.test_session() as sess:
      output, gradients = sess.run([output, gradients])
      self.assertFalse(np.any(np.isnan(output)))
      self.assertFalse(np.any(np.isnan(gradients[0])))
      self.assertFalse(np.any(np.isnan(gradients[1])))
      self.assertFalse(np.any(np.isnan(gradients[2])))


class TemporalLinkageTest(tf.test.TestCase):

  def testModule(self):
    batch_size = 7
    memory_size = 4
    num_reads = 11
    num_writes = 5
    module = addressing.TemporalLinkage(
        memory_size=memory_size, num_writes=num_writes)

    prev_link_in = tf.placeholder(
        tf.float32, (batch_size, num_writes, memory_size, memory_size))
    prev_precedence_weights_in = tf.placeholder(
        tf.float32, (batch_size, num_writes, memory_size))
    write_weights_in = tf.placeholder(tf.float32,
                                      (batch_size, num_writes, memory_size))

    state = addressing.TemporalLinkageState(
        link=np.zeros([batch_size, num_writes, memory_size, memory_size]),
        precedence_weights=np.zeros([batch_size, num_writes, memory_size]))

    calc_state = module(write_weights_in,
                        addressing.TemporalLinkageState(
                            link=prev_link_in,
                            precedence_weights=prev_precedence_weights_in))

    with self.test_session() as sess:
      num_steps = 5
      for i in xrange(num_steps):
        write_weights = np.random.rand(batch_size, num_writes, memory_size)
        write_weights /= write_weights.sum(2, keepdims=True) + 1

        # Simulate (in final steps) link 0-->1 in head 0 and 3-->2 in head 1
        if i == num_steps - 2:
          write_weights[0, 0, :] = util.one_hot(memory_size, 0)
          write_weights[0, 1, :] = util.one_hot(memory_size, 3)
        elif i == num_steps - 1:
          write_weights[0, 0, :] = util.one_hot(memory_size, 1)
          write_weights[0, 1, :] = util.one_hot(memory_size, 2)

        state = sess.run(
            calc_state,
            feed_dict={
                prev_link_in: state.link,
                prev_precedence_weights_in: state.precedence_weights,
                write_weights_in: write_weights
            })

    # link should be bounded in range [0, 1]
    self.assertGreaterEqual(state.link.min(), 0)
    self.assertLessEqual(state.link.max(), 1)

    # link diagonal should be zero
    self.assertAllEqual(
        state.link[:, :, range(memory_size), range(memory_size)],
        np.zeros([batch_size, num_writes, memory_size]))

    # link rows and columns should sum to at most 1
    self.assertLessEqual(state.link.sum(2).max(), 1)
    self.assertLessEqual(state.link.sum(3).max(), 1)

    # records our transitions in batch 0: head 0: 0->1, and head 1: 3->2
    self.assertAllEqual(state.link[0, 0, :, 0], util.one_hot(memory_size, 1))
    self.assertAllEqual(state.link[0, 1, :, 3], util.one_hot(memory_size, 2))

    # Now test calculation of forward and backward read weights
    prev_read_weights = np.random.rand(batch_size, num_reads, memory_size)
    prev_read_weights[0, 5, :] = util.one_hot(memory_size, 0)  # read 5, posn 0
    prev_read_weights[0, 6, :] = util.one_hot(memory_size, 2)  # read 6, posn 2
    forward_read_weights = module.directional_read_weights(
        tf.constant(state.link),
        tf.constant(prev_read_weights, dtype=tf.float32),
        forward=True)
    backward_read_weights = module.directional_read_weights(
        tf.constant(state.link),
        tf.constant(prev_read_weights, dtype=tf.float32),
        forward=False)

    with self.test_session():
      forward_read_weights = forward_read_weights.eval()
      backward_read_weights = backward_read_weights.eval()

    # Check directional weights calculated correctly.
    self.assertAllEqual(
        forward_read_weights[0, 5, 0, :],  # read=5, write=0
        util.one_hot(memory_size, 1))
    self.assertAllEqual(
        backward_read_weights[0, 6, 1, :],  # read=6, write=1
        util.one_hot(memory_size, 3))

  def testPrecedenceWeights(self):
    batch_size = 7
    memory_size = 3
    num_writes = 5
    module = addressing.TemporalLinkage(
        memory_size=memory_size, num_writes=num_writes)

    prev_precedence_weights = np.random.rand(batch_size, num_writes,
                                             memory_size)
    write_weights = np.random.rand(batch_size, num_writes, memory_size)

    # These should sum to at most 1 for each write head in each batch.
    write_weights /= write_weights.sum(2, keepdims=True) + 1
    prev_precedence_weights /= prev_precedence_weights.sum(2, keepdims=True) + 1

    write_weights[0, 1, :] = 0  # batch 0 head 1: no writing
    write_weights[1, 2, :] /= write_weights[1, 2, :].sum()  # b1 h2: all writing

    precedence_weights = module._precedence_weights(
        prev_precedence_weights=tf.constant(prev_precedence_weights),
        write_weights=tf.constant(write_weights))

    with self.test_session():
      precedence_weights = precedence_weights.eval()

    # precedence weights should be bounded in range [0, 1]
    self.assertGreaterEqual(precedence_weights.min(), 0)
    self.assertLessEqual(precedence_weights.max(), 1)

    # no writing in batch 0, head 1
    self.assertAllClose(precedence_weights[0, 1, :],
                        prev_precedence_weights[0, 1, :])

    # all writing in batch 1, head 2
    self.assertAllClose(precedence_weights[1, 2, :], write_weights[1, 2, :])


class FreenessTest(tf.test.TestCase):

  def testModule(self):
    batch_size = 5
    memory_size = 11
    num_reads = 3
    num_writes = 7
    module = addressing.Freeness(memory_size)

    free_gate = np.random.rand(batch_size, num_reads)

    # Produce read weights that sum to 1 for each batch and head.
    prev_read_weights = np.random.rand(batch_size, num_reads, memory_size)
    prev_read_weights[1, :, 3] = 0  # no read at batch 1, position 3; see below
    prev_read_weights /= prev_read_weights.sum(2, keepdims=True)
    prev_write_weights = np.random.rand(batch_size, num_writes, memory_size)
    prev_write_weights /= prev_write_weights.sum(2, keepdims=True)
    prev_usage = np.random.rand(batch_size, memory_size)

    # Add some special values that allows us to test the behaviour:
    prev_write_weights[1, 2, 3] = 1  # full write in batch 1, head 2, position 3
    prev_read_weights[2, 0, 4] = 1  # full read at batch 2, head 0, position 4
    free_gate[2, 0] = 1  # can free up all locations for batch 2, read head 0

    usage = module(
        tf.constant(prev_write_weights),
        tf.constant(free_gate),
        tf.constant(prev_read_weights), tf.constant(prev_usage))
    with self.test_session():
      usage = usage.eval()

    # Check all usages are between 0 and 1.
    self.assertGreaterEqual(usage.min(), 0)
    self.assertLessEqual(usage.max(), 1)

    # Check that the full write at batch 1, position 3 makes it fully used.
    self.assertEqual(usage[1][3], 1)

    # Check that the full free at batch 2, position 4 makes it fully free.
    self.assertEqual(usage[2][4], 0)

  def testWriteAllocationWeights(self):
    batch_size = 7
    memory_size = 23
    num_writes = 5
    module = addressing.Freeness(memory_size)

    usage = np.random.rand(batch_size, memory_size)
    write_gates = np.random.rand(batch_size, num_writes)

    # Turn off gates for heads 1 and 3 in batch 0. This doesn't scaling down the
    # weighting, but it means that the usage doesn't change, so we should get
    # the same allocation weightings for: (1, 2) and (3, 4) (but all others
    # being different).
    write_gates[0, 1] = 0
    write_gates[0, 3] = 0
    # and turn heads 0 and 2 on for full effect.
    write_gates[0, 0] = 1
    write_gates[0, 2] = 1

    # In batch 1, make one of the usages 0 and another almost 0, so that these
    # entries get most of the allocation weights for the first and second heads.
    usage[1] = usage[1] * 0.9 + 0.1  # make sure all entries are in [0.1, 1]
    usage[1][4] = 0  # write head 0 should get allocated to position 4
    usage[1][3] = 1e-4  # write head 1 should get allocated to position 3
    write_gates[1, 0] = 1  # write head 0 fully on
    write_gates[1, 1] = 1  # write head 1 fully on

    weights = module.write_allocation_weights(
        usage=tf.constant(usage),
        write_gates=tf.constant(write_gates),
        num_writes=num_writes)

    with self.test_session():
      weights = weights.eval()

    # Check that all weights are between 0 and 1
    self.assertGreaterEqual(weights.min(), 0)
    self.assertLessEqual(weights.max(), 1)

    # Check that weights sum to close to 1
    self.assertAllClose(
        np.sum(weights, axis=2), np.ones([batch_size, num_writes]), atol=1e-3)

    # Check the same / different allocation weight pairs as described above.
    self.assertGreater(np.abs(weights[0, 0, :] - weights[0, 1, :]).max(), 0.1)
    self.assertAllEqual(weights[0, 1, :], weights[0, 2, :])
    self.assertGreater(np.abs(weights[0, 2, :] - weights[0, 3, :]).max(), 0.1)
    self.assertAllEqual(weights[0, 3, :], weights[0, 4, :])

    self.assertAllClose(weights[1][0], util.one_hot(memory_size, 4), atol=1e-3)
    self.assertAllClose(weights[1][1], util.one_hot(memory_size, 3), atol=1e-3)

  def testWriteAllocationWeightsGradient(self):
    batch_size = 7
    memory_size = 5
    num_writes = 3
    module = addressing.Freeness(memory_size)

    usage = tf.constant(np.random.rand(batch_size, memory_size))
    write_gates = tf.constant(np.random.rand(batch_size, num_writes))
    weights = module.write_allocation_weights(usage, write_gates, num_writes)

    with self.test_session():
      err = tf.test.compute_gradient_error(
          [usage, write_gates],
          [usage.get_shape().as_list(), write_gates.get_shape().as_list()],
          weights,
          weights.get_shape().as_list(),
          delta=1e-5)
      self.assertLess(err, 0.01)

  def testAllocation(self):
    batch_size = 7
    memory_size = 13
    usage = np.random.rand(batch_size, memory_size)
    module = addressing.Freeness(memory_size)
    allocation = module._allocation(tf.constant(usage))
    with self.test_session():
      allocation = allocation.eval()

    # 1. Test that max allocation goes to min usage, and vice versa.
    self.assertAllEqual(np.argmin(usage, axis=1), np.argmax(allocation, axis=1))
    self.assertAllEqual(np.argmax(usage, axis=1), np.argmin(allocation, axis=1))

    # 2. Test that allocations sum to almost 1.
    self.assertAllClose(np.sum(allocation, axis=1), np.ones(batch_size), 0.01)

  def testAllocationGradient(self):
    batch_size = 1
    memory_size = 5
    usage = tf.constant(np.random.rand(batch_size, memory_size))
    module = addressing.Freeness(memory_size)
    allocation = module._allocation(usage)
    with self.test_session():
      err = tf.test.compute_gradient_error(
          usage,
          usage.get_shape().as_list(),
          allocation,
          allocation.get_shape().as_list(),
          delta=1e-5)
      self.assertLess(err, 0.01)


if __name__ == '__main__':
  tf.test.main()

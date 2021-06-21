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
"""Tests for DNCCore"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import random_seed

from dnc import dnc, access, addressing
from dnc import repeat_copy

# set seeds for determinism
np.random.seed(42)
random_seed.set_seed(42)

DTYPE = tf.float32

# Model parameters
HIDDEN_SIZE = 64
MEMORY_SIZE = 16
WORD_SIZE = 16
NUM_WRITE_HEADS = 1
NUM_READ_HEADS = 4
CLIP_VALUE = 20

# Optimizer parameters.
MAX_GRAD_NORM = 50
LEARNING_RATE = 1e-4
OPTIMIZER_EPSILON = 1e-10

# Task parameters
BATCH_SIZE = 16
TIME_STEPS = 4
INPUT_SIZE = 4
OUTPUT_SIZE = 4


class DNCCoreTest(tf.test.TestCase):
    def setUp(self):
        access_config = {
            "memory_size": MEMORY_SIZE,
            "word_size": WORD_SIZE,
            "num_reads": NUM_READ_HEADS,
            "num_writes": NUM_WRITE_HEADS,
        }
        controller_config = {
            # "hidden_size": FLAGS.hidden_size,
            "units": HIDDEN_SIZE,
        }

        self.module = dnc.DNC(
            access_config,
            controller_config,
            OUTPUT_SIZE,
            BATCH_SIZE,
            CLIP_VALUE,
            name="dnc_test",
            dtype=DTYPE,
        )
        self.initial_state = self.module.get_initial_state(batch_size=BATCH_SIZE)

    def testBuildAndTrain(self):
        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE], dtype=DTYPE)
        targets = np.random.rand(TIME_STEPS, BATCH_SIZE, OUTPUT_SIZE)

        def loss(outputs, targets):
            return tf.reduce_mean(input_tensor=tf.square(outputs - targets))

        optimizer = tf.compat.v1.train.RMSPropOptimizer(
            LEARNING_RATE, epsilon=OPTIMIZER_EPSILON
        )

        with tf.GradientTape() as tape:
            # outputs, _ = tf.compat.v1.nn.dynamic_rnn(
            outputs = tf.keras.layers.RNN(
                cell=self.module,
                time_major=True,
                return_sequences=True,
            )(
                inputs=inputs,
                initial_state=self.initial_state,
            )
            loss_value = loss(outputs, targets)
            gradients = tape.gradient(loss_value, self.module.trainable_variables)

        grads, _ = tf.clip_by_global_norm(gradients, MAX_GRAD_NORM)
        optimizer.apply_gradients(zip(gradients, self.module.trainable_variables))

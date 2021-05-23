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
"""DNC Cores.

These modules create a DNC core. They take input, pass parameters to the memory
access module, and integrate the output of memory to form an output.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import sonnet as snt
import tensorflow as tf

from dnc import access, util

DNCState = collections.namedtuple('DNCState',
    ('access_output', 'access_state', 'controller_state'))


class DNC(snt.RNNCore):
  """DNC core module.

  Contains controller and memory access module.
  """

  def __init__(self,
               access_config,
               controller_config,
               output_size,
               batch_size,
               clip_value=None,
               name='dnc',
               dtype=tf.float32):
    """Initializes the DNC core.

    Args:
      access_config: dictionary of access module configurations.
      controller_config: dictionary of controller (LSTM) module configurations.
      output_size: output dimension size of core.
      clip_value: clips controller and core output values to between
          `[-clip_value, clip_value]` if specified.
      name: module name (default 'dnc').

    Raises:
      TypeError: if direct_input_size is not None for any access module other
        than KeyValueMemory.
    """
    super(DNC, self).__init__(name=name)

    self._dtype = dtype
    # dm-sonnet=2.0.0 LSTM is not integrated with TF2 tracing.
    #   Use keras to allow for Tensorboard visualization
    #self._controller = snt.LSTM(**controller_config, dtype=tf.float64)
    self._controller = tf.keras.layers.LSTMCell(**controller_config, dtype=dtype)
    self._access = access.MemoryAccess(**access_config, dtype=dtype)

    self._access_output_size = np.prod(self._access.output_size.as_list())
    self._output_size = output_size
    self._batch_size = batch_size
    self._clip_value = clip_value or 0

    self._output_size = tf.TensorShape([output_size])
    self._state_size = DNCState(
        access_output=self._access_output_size,
        access_state=self._access.state_size,
        controller_state=self._controller.state_size,
    )
    self._output_linear = snt.Linear(
        output_size=self._output_size.as_list()[0],
        name='output_linear')

  def _clip_if_enabled(self, x):
    if self._clip_value > 0:
      return tf.clip_by_value(x, -self._clip_value, self._clip_value)
    else:
      return x

  def __call__(self, inputs, prev_state):
    """Connects the DNC core into the graph.

    Args:
      inputs: Tensor input.
      prev_state: A `DNCState` tuple containing the fields `access_output`,
          `access_state` and `controller_state`. `access_state` is a 3-D Tensor
          of shape `[batch_size, num_reads, word_size]` containing read words.
          `access_state` is a tuple of the access module's state, and
          `controller_state` is a tuple of controller module's state.

    Returns:
      A tuple `(output, next_state)` where `output` is a tensor and `next_state`
      is a `DNCState` tuple containing the fields `access_output`,
      `access_state`, and `controller_state`.
    """
    prev_access_output = prev_state.access_output
    prev_access_state = prev_state.access_state
    prev_controller_state = prev_state.controller_state

    batch_flatten = tf.keras.layers.Flatten()
    controller_input = tf.concat(
        [batch_flatten(inputs), batch_flatten(prev_access_output)], 1)

    controller_output, controller_state = self._controller(
        controller_input, prev_controller_state)

    controller_output = self._clip_if_enabled(controller_output)
    controller_state = tf.nest.map_structure(self._clip_if_enabled, controller_state)

    access_output, access_state = self._access(controller_output,
                                               prev_access_state)

    output = tf.concat([controller_output, batch_flatten(access_output)], 1)
    output = self._output_linear(output)
    output = self._clip_if_enabled(output)

    return output, DNCState(
        access_output=access_output,
        access_state=access_state,
        controller_state=controller_state)

  def initial_state(self, batch_size=None):
    return self.get_initial_state(batch_size)

  def get_initial_state(self, batch_size=None):
    return DNCState(
        controller_state=self._controller.get_initial_state(batch_size=batch_size, dtype=self._dtype),
        access_state=self._access.get_initial_state(batch_size),
        access_output=tf.zeros(
            [batch_size] + self._access.output_size.as_list(), dtype=self._dtype))

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

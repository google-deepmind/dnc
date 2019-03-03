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

from dnc import access

DNCState = collections.namedtuple('DNCState', ('access_output', 'access_state'))


class DNCfeedforward(snt.RNNCore):
  """DNC core module.

  Contains controller and memory access module.
  """

  def __init__(self,
               access_config,
               controller_config,
               output_size,
               clip_value=None,
               name='dnc',
               return_weights=False):
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
    super(DNCfeedforward, self).__init__(name=name)

    self._memory_size = access_config.get('memory_size')
    self._nb_extra_output = 2 * self._memory_size

    with self._enter_variable_scope():
      self._controller = snt.Linear(
        output_size=controller_config.get('hidden_size'),
        name='controller'
      ) #snt.LSTM(**controller_config)
      self._access = access.MemoryAccess(**access_config)

    self._access_output_size = np.prod(self._access.output_size.as_list())
    if return_weights:
       output_size += self._nb_extra_output  # TODO make this memory size

    print('output size:')
    print(output_size)

    self._output_size = output_size

    self._clip_value = clip_value or 0

    self._output_size = tf.TensorShape([output_size])
    self._state_size = DNCState(
        access_output=self._access_output_size,
        access_state=self._access.state_size)

    self._return_weights = return_weights

  def _clip_if_enabled(self, x):
    if self._clip_value > 0:
      return tf.clip_by_value(x, -self._clip_value, self._clip_value)
    else:
      return x

  def _build(self, inputs, prev_state):
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

    batch_flatten = snt.BatchFlatten()
    controller_input = tf.concat(
        [batch_flatten(inputs), batch_flatten(prev_access_output)], 1)

    controller_output = self._controller(
        controller_input)

    # Controller output: (batch_size, hidden_size) tensor.

    controller_output = self._clip_if_enabled(controller_output)

    access_output, access_state = self._access(controller_output,
                                               prev_access_state)

    output = tf.concat([controller_output, batch_flatten(access_output)], 1)
    output_size = self._output_size.as_list()[0]
    if self._return_weights:
      output_size -= self._nb_extra_output # TODO extract into variable

    output = snt.Linear(
        output_size=output_size,
        name='output_linear')(output)
    output = self._clip_if_enabled(output)

    write_weights = access_state.write_weights
    read_weights = access_state.read_weights

    if self._return_weights:
      output = tf.concat([output, read_weights[:, 0, :], write_weights[:, 0, :]], 1)

    return output, DNCState(
        access_output=access_output,
        access_state=access_state)

  def initial_state(self, batch_size, dtype=tf.float32):
    return DNCState(
        access_state=self._access.initial_state(batch_size, dtype),
        access_output=tf.zeros(
            [batch_size] + self._access.output_size.as_list(), dtype))

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

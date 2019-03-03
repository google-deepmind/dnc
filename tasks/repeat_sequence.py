import sonnet as snt
import tensorflow as tf

import random
import collections
import numpy as np

DatasetTensors = collections.namedtuple('DatasetTensors', ('observations', 'target'))

# Repeat a sequence one time. Could at more sequences later for testing
class RepeatSequence(snt.AbstractModule):

  def __init__(self,
               min_nb_vecs=3,
               max_nb_vecs=5,
               nb_bits=7,
               batch_size=16):
    super(RepeatSequence, self).__init__(name='RepeatSequence')

    self._min_nb_vecs = min_nb_vecs
    self._max_nb_vecs = max_nb_vecs
    self._nb_bits = nb_bits
    self._batch_size = batch_size

    self.target_size = self._nb_bits + 1

  def _build(self):
    obs_tensors = []
    target_tensors = []

    # Create the random variable
    nb_vecs_batch = tf.random_uniform(
        [1], minval=self._min_nb_vecs, maxval=self._max_nb_vecs+1, dtype=tf.int32)

    for batch_index in range(0, self._batch_size):
      nb_vecs = nb_vecs_batch[0]

      # OBSERVATION

      # Pattern
      obs_pattern = tf.cast(
        tf.random_uniform([nb_vecs, self._nb_bits], minval=0, maxval=2, dtype=tf.int32),
        tf.float32
      )

      # Add zeros in the flag channel
      obs_flag_channel_zeros = tf.zeros([nb_vecs, 1], dtype=tf.float32)
      obs = tf.concat([obs_flag_channel_zeros, obs_pattern], 1)

      # Add the flag to the observation
      observation_with_flag = tf.concat([
        obs,
        tf.expand_dims(tf.concat([tf.constant([1], dtype=tf.float32), tf.zeros(self._nb_bits, dtype=tf.float32)], 0), 0)
      ], 0)

      # Pad the observation
      observation_padded = tf.concat([
        observation_with_flag,
        tf.zeros([nb_vecs, self.target_size], dtype=tf.float32)
      ], 0)

      # TARGET

      # Add zeros (padding) to the target
      target_padding = tf.zeros([nb_vecs + 1, self.target_size], dtype=tf.float32)

      # Add the observation to the target
      target = tf.concat([target_padding, obs], 0)

      obs_tensors.append(observation_padded)
      target_tensors.append(target)

    return DatasetTensors(tf.convert_to_tensor(obs_tensors), tf.convert_to_tensor(target_tensors))

  def time_major(self):
    return False

  def to_human_readable(self, data, model_output=None, whole_batch=False):
    return bitstring_readable(data, self._batch_size, model_output, whole_batch)

  def cost(self, logits, targ):
    xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=targ, logits=logits)

    # Sum across the vectors
    loss_batch_time = tf.reduce_sum(xent, axis=2)

    # Sum away time
    loss_batch = tf.reduce_sum(loss_batch_time, axis=1)

    # Batch major
    batch_size = tf.cast(tf.shape(logits)[0], dtype=loss_batch_time.dtype)
    loss = tf.reduce_sum(loss_batch) / batch_size

    return loss


def bitstring_readable(data, batch_size, model_output=None, whole_batch=False):
  """Produce a human readable representation of the sequences in data.

  Args:
    data: data to be visualised
    batch_size: size of batch
    model_output: optional model output tensor to visualize alongside data.
    whole_batch: whether to visualise the whole batch. Only the first sample
        will be visualized if False

  Returns:
    A string used to visualise the data batch. X axis is time, Y axis are vectors
  """
  def _readable(datum):
    return '+' + ' '.join(['-' if x == 0 else '%d' % x for x in datum]) + '+'


  # Transform input: shape is (batch_size, max_time, nb_bits)
  # This must become (batch_size, nb_bits, max_time)

  obs_batch = data.observations.transpose((0, 2, 1))
  targ_batch = data.target.transpose((0, 2, 1))

  iterate_over = range(batch_size) if whole_batch else range(1)

  batch_strings = []

  for batch_index in iterate_over:
    obs = obs_batch[batch_index, :, :]
    targ = targ_batch[batch_index, :, :]

    readable_obs = 'Observations:\n' + '\n'.join([_readable(obs_vector) for obs_vector in obs])
    readable_targ = 'Targets:\n' + '\n'.join([_readable(targ_vector) for targ_vector in targ])
    strings = [readable_obs, readable_targ]

    if model_output is not None:
      output = model_output.transpose((0, 2, 1))[batch_index, :, :]
      strings.append('Model Output:\n' + '\n'.join([_readable(output_vec) for output_vec in output]))

    batch_strings.append('\n\n'.join(strings))

  return '\n' + '\n\n\n\n'.join(batch_strings)
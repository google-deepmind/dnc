import collections
import numpy as np
import sonnet as snt
import tensorflow as tf

DatasetTensors = collections.namedtuple('DatasetTensors', ('observations', 'target'))

class Copy4Data(snt.AbstractModule):

  def __init__(self, num_bits=10, batch_size=16, name='copy_4'):
    super(Copy4Data, self).__init__(name=name)
    self._num_bits = num_bits
    self._batch_size = batch_size

  def _build(self):
    obs_tensors = []
    target_tensors = []

    for batch_index in range(0, self._batch_size):
      obs_pattern = tf.cast(
        tf.random_uniform(
          [10], minval=0, maxval=2, dtype=tf.int32),
        tf.float32)

      targ_pattern = tf.reshape(tf.tile(obs_pattern, [4]), [4, 10])

      obs_pad = tf.zeros([5, 10])
      targ_pad = tf.zeros([2, 10])

      obs = tf.concat([tf.reshape(obs_pattern, [1, 10]), obs_pad], axis=0)
      targ = tf.concat([targ_pad, targ_pattern], axis=0)

      obs_tensors.append(obs)
      target_tensors.append(targ)

    return DatasetTensors(tf.convert_to_tensor(obs_tensors), tf.convert_to_tensor(target_tensors))

  # TODO add the time_average and log_prob_in_bits if required
  # def cost(self, logits, targ):
    # xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=targ, logits=logits)
    # loss_time_batch = tf.reduce_sum(xent, axis=2)
    # loss_batch = tf.reduce_sum(loss_time_batch, axis=0)
    #
    # batch_size = tf.cast(tf.shape(logits)[1], dtype=loss_time_batch.dtype)
    #
    # loss = tf.reduce_sum(loss_batch) / batch_size
    #
    # return loss
  # TODO this cost funciton is not right

  def cost(self, logits, targ):
    xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=targ, logits=logits)
    loss_time_batch = tf.reduce_sum(xent, axis=2)
    loss_batch = tf.reduce_sum(loss_time_batch, axis=1)

    batch_size = tf.cast(tf.shape(logits)[0], dtype= loss_time_batch.dtype)

    loss = tf.reduce_sum(loss_batch) / batch_size

    return loss

  def to_human_readable(self, data, model_output=None, whole_batch=False):
    return bitstring_readable(data, self._batch_size, model_output, whole_batch)

def bitstring_readable(data, batch_size, model_output=None, whole_batch=False):
  """Produce a human readable representation of the sequences in data.

  Args:
    data: data to be visualised
    batch_size: size of batch
    model_output: optional model output tensor to visualize alongside data.
    whole_batch: whether to visualise the whole batch. Only the first sample
        will be visualized if False

  Returns:
    A string used to visualise the data batch
  """

  def _readable(datum):
    return '+' + ' '.join(['-' if x == 0 else '%d' % x for x in datum]) + '+'

  obs_batch = data.observations
  targ_batch = data.target

  iterate_over = range(batch_size) if whole_batch else range(1)

  batch_strings = []

  for batch_index in iterate_over:
    obs = obs_batch[batch_index, :, :]
    targ = targ_batch[batch_index, :, :]

    readable_obs = 'Observations:\n' + '\n'.join([_readable(obs_vector) for obs_vector in obs])
    readable_targ = 'Targets:\n' + '\n'.join([_readable(targ_vector) for targ_vector in targ])
    strings = [readable_obs, readable_targ]

    if model_output is not None:
      output = model_output[batch_index, :, :]
      strings.append('Model Output:\n' + '\n'.join([_readable(output_vec) for output_vec in output]))

    batch_strings.append('\n\n'.join(strings))

  return '\n' + '\n\n\n\n'.join(batch_strings)

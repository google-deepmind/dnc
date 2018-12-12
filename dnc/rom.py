import tensorflow as tf
import sonnet as snt

import addressing

# TODO:
# - 'Read' the rom output somewhere: extract mu, address
# - Check the initialisation of the state of the rom and mixer
# - Check the dimensions of the read key: make it match word size.
# - Right now, 'linkage' is just the next element in memory. Implement it like the DNC to allow multiple next elements.
# - Make the mixer more extensible

class ROM(snt.RNNCore):
  def __init__(self,
               content,
               name='ROM'):
    super(ROM, self).__init__(name=name)
    self._content = content
    self._memory_size = tf.shape(content)[0]  # make one op of these two
    self._word_size = tf.shape(content)[1]

    self._read_content_weight_mod = addressing.CosineWeights(
        1, self._word_size, name='read_content_weights')

  # Recurrent, because readings of the mode depend on the previous read weight
  # Input is batch major
  # read_mode: 2 elements, first is content, second is forward
  def _build(self, read_key, read_strength, read_mode, prev_read_weight):
    # Tile content for batch: content is the same for each element of the batch:
    batch_size = tf.shape(read_key)[0]
    content = tf.tile(tf.expand_dims(self._content, 0), [batch_size, 1, 1])
    # Expand dims is because the cosineweights is defined for multiple heads at once but we only have 1
    content_weights = self._read_content_weight_mod(
      content, tf.expand_dims(read_key, 1), tf.expand_dims(read_strength, 1)
    )[:, 0, :]

    forward_weight = self._forward_read_weight(prev_read_weight)

    # Repeat the mode to be able to multiply it with every element of the weights
    content_mode = tf.tile(tf.expand_dims(read_mode[:, 0], 1), [1, self._memory_size])
    forward_mode = tf.tile(tf.expand_dims(read_mode[:, 1], 1), [1, self._memory_size])

    read_weight = tf.multiply(content_mode, content_weights) + tf.multiply(forward_mode, forward_weight)
    read_word = tf.matmul(read_weight, self._content)

    return read_word, read_weight

  # Expects that input is batch major
  def _forward_read_weight(self, prev_read_weights):
    return tf.manip.roll(prev_read_weights, shift=1, axis=1)

class Mixer(snt.RNNCore):
  def __init__(self, name='Mixer'):
    super(Mixer, self).__init__(name=name)

  # Recurrent, because mu depends on previous mu.
  # Returns the mixed value.
  # 1-D vectors for mu: [batch_size].
  def _build(self, value_controller, value_rom, mu_controller, mu_rom, prev_mu):
    batch_size = tf.shape(mu_controller)[0]
    word_size = tf.shape(value_controller)[1]

    # mu = prev_mu * mu_rom + (1 - prev_mu) * mu_controller
    mu = tf.add(
      tf.multiply(prev_mu, mu_rom),
      tf.multiply(tf.subtract(tf.tile(tf.constant([1], dtype='float32'), [batch_size]), prev_mu), mu_controller))

    # value = mu * value_rom + (1 - mu) * value_controller
    tiled_mu = tf.tile(tf.expand_dims(mu, 1), [1, word_size])
    value = tf.add(
      tf.multiply(tiled_mu, value_rom),
      tf.multiply(tf.subtract(tf.ones([batch_size, word_size], dtype='float32'), tiled_mu), value_controller))

    return value, mu


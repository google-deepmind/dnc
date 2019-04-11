import tensorflow as tf
import sonnet as snt

from dnc import addressing
# import addressing

# TODO:
# - 'Read' the rom output somewhere: extract mu, address
# - Check the initialisation of the state of the rom and mixer
# - Right now, 'linkage' is just the next element in memory. Implement it like the DNC to allow multiple next elements.
# - Make the mixer more extensible

class ROM(snt.AbstractModule):
  def __init__(self,
               content,
               key_size,
               name='ROM'):
    super(ROM, self).__init__(name=name)
    self._content = content
    self._memory_size = tf.shape(content)[0]  # make one op of these two
    self._key_size = key_size

    self._read_content_weight_mod = addressing.CosineWeights(
        1, key_size, name='read_content_weights')

  # Recurrent, because readings of the mode depend on the previous read weight
  # Input is batch major
  # read_mode: 2 elements, first is content, second is forward
  def _build(self, read_key, read_strength, read_mode, prev_read_weight):
    # Tile content for batch: content is the same for each element of the batch:
    batch_size = tf.shape(read_key)[0]
    content = tf.tile(tf.expand_dims(self._content, 0), [batch_size, 1, 1])
    # Expand dims is because the cosineweights is defined for multiple heads at once but we only have 1
    content_weight = self._read_content_weight_mod(
      content[:, :, 0:self._key_size], tf.expand_dims(read_key, 1), read_strength   # TODO check that this content indices are correct
    )[:, 0, :]

    forward_weight = self._forward_read_weight(prev_read_weight)

    # Repeat the mode to be able to multiply it with every element of the weights
    content_mode = tf.tile(tf.expand_dims(read_mode[:, 0], 1), [1, self._memory_size])
    forward_mode = tf.tile(tf.expand_dims(read_mode[:, 1], 1), [1, self._memory_size])

    read_weight = tf.multiply(content_mode, content_weight) + tf.multiply(forward_mode, forward_weight)
    read_word = tf.matmul(read_weight, self._content)

    return read_word, read_weight

  # Expects that input is batch major
  def _forward_read_weight(self, prev_read_weights):
    return tf.manip.roll(prev_read_weights, shift=1, axis=1)

  def word_size(self):
    return self._content.get_shape().as_list()[1]

  def key_size(self):
    return self._key_size

  def memory_size(self):
    return self._content.get_shape().as_list()[0]

class Mixer(snt.AbstractModule):
  def __init__(self, name='Mixer'):
    super(Mixer, self).__init__(name=name)

  # mu is expected to be a 1d array of length batch_size x 1
  # value_controller and value_rom expected to be batch_size x l
  # the used parameter specifies if the value is used on the rom. 1d array of length batch_size x 1
  def _build(self, value_controller, value_rom, mu, used):
    batch_size = tf.shape(mu)[0]
    word_size = tf.shape(value_controller)[1]
    print('Word size: ')
    print(word_size)
    print('Mu: ')
    print(mu)
    tiled_mu = tf.tile(mu, [1, word_size])
    tiled_used = tf.tile(used, [1, word_size])

    z = tiled_mu * tiled_used # Can experiment with different functions here

    # value = mu * value_rom + (1 - mu) * value_controller
    value = tf.add(
      tf.multiply(z, value_rom),
      tf.multiply(tf.subtract(tf.ones([batch_size, word_size], dtype='float32'), z), value_controller)
    )

    return value

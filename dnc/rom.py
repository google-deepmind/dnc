import tensorflow as tf
import sonnet as snt

from dnc import addressing
# import addressing

# TODO:
# - Right now, 'linkage' is just the next element in memory. Implement it like the DNC to allow multiple next elements.
# - Make the mixer more extensible


class ROM(snt.AbstractModule):
  def __init__(self,
               content,
               memory_size,
               name='ROM'):
    super(ROM, self).__init__(name=name)

    self._defaults = {
      'allocation_gate': 1,
      'write_gate': 1,
      'read_mode': 3,
      'read_weight': memory_size,
      'write_weight': memory_size,
      'next_rom_mode': 2
    }
    self._content = self.read_content(content, self._defaults, 'float32')
    self._rom_memory_size = len(content)
    self._key_size = len(content[0][0])
    self._read_content_weight_mod = addressing.CosineWeights(
        1, self._key_size)

  def read_content(self, content, defaults, dtype):
    result = {
      'key': tf.constant([c[0] for c in content], dtype=dtype),
      'mu': tf.constant([c[2] for c in content], dtype=dtype)
    }
    dicts = [c[1] for c in content]
    for key in defaults.keys():
      default = [0] * (defaults[key] + 1) # 1 is for the usage
      result[key] = tf.constant([self.read_dict_key(dict, key, default) for dict in dicts], dtype=dtype)

    return result

  def read_dict_key(self, dict, key, default):
    value = dict.get(key)
    return [1] + value if value else default

  # Recurrent, because readings of the mode depend on the previous read weight
  # Input is batch major
  # read_mode: 2 elements, first is content, second is forward
  def _build(self, read_key, read_strength, read_mode, prev_read_weight):
    # Tile keys for batch: content is the same for each element of the batch:
    batch_size = tf.shape(read_key)[0]
    keys = tf.tile(tf.expand_dims(self._content['key'], 0), [batch_size, 1, 1])
    # Expand dims is because the cosineweights is defined for multiple heads at once but we only have 1
    content_weight = self._read_content_weight_mod(
      keys, tf.expand_dims(read_key, 1), read_strength
    )[:, 0, :]

    forward_weight = self._forward_read_weight(prev_read_weight)

    # Repeat the mode to be able to multiply it with every element of the weights
    content_mode = tf.tile(tf.expand_dims(read_mode[:, 0], 1), [1, self._rom_memory_size])
    forward_mode = tf.tile(tf.expand_dims(read_mode[:, 1], 1), [1, self._rom_memory_size])

    read_weight = tf.multiply(content_mode, content_weight) + tf.multiply(forward_mode, forward_weight)
    read_word = self.read_rom(read_weight)

    return read_word, read_weight

  # Computes a weighted sum of usage and the content for each of the keys
  # read_weight: batch_size x rom_memory_size
  # result: a dict where every element is batch_size x ? (? = the size of the content on that key + 1)
  def read_rom(self, read_weight):
    result = {
      'mu': tf.matmul(read_weight, tf.expand_dims(self._content['mu'], 1))
    }
    for key in self._defaults.keys():
      usages = self._content[key][:, 0] # length rom_mem_size
      values = self._content[key][:, 1:]
      dont_care_weighting = read_weight * usages # batch_size x rom_mem_size
      resulting_usage = tf.reduce_sum(dont_care_weighting, 1) # length batch_size
      normalized_dont_care_weighting = tf.div_no_nan(dont_care_weighting, tf.tile(tf.expand_dims(resulting_usage, 1), [1, self._rom_memory_size]))
      read_val = tf.matmul(normalized_dont_care_weighting, values) # batch_size x ?
      result[key] = tf.concat([tf.expand_dims(resulting_usage, 1),read_val], 1)

    return result

  # Expects that input is batch major
  def _forward_read_weight(self, prev_read_weights):
    return tf.manip.roll(prev_read_weights, shift=1, axis=1)

  def key_size(self):
    return self._key_size

  def memory_size(self):
    return self._rom_memory_size

  def create_content(self, key, content, mu):
    return self._content_factory.create_content(key, content, mu)


class Mixer(snt.AbstractModule):
  def __init__(self, name='Mixer'):
    super(Mixer, self).__init__(name=name)

  # mu is expected to be a 1d array of length batch_size x 1
  # value_controller and value_rom expected to be batch_size x l
  # the used parameter specifies if the value is used on the rom. 1d array of length batch_size x 1
  def _build(self, value_controller, value_rom, mu, used):
    batch_size = tf.shape(mu)[0]
    word_size = tf.shape(value_controller)[1]

    tiled_mu = tf.tile(mu, [1, word_size])
    tiled_used = tf.tile(used, [1, word_size])

    z = tiled_mu * tiled_used # Can experiment with different functions here

    # value = z * value_rom + (1 - z) * value_controller
    value = tf.add(
      tf.multiply(z, value_rom),
      tf.multiply(tf.subtract(tf.ones([batch_size, word_size], dtype='float32'), z), value_controller)
    )

    return value

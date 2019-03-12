import tensorflow as tf

class ROMContentFactory:
  """
  content: dict of values to be put on the rom
  address_length: int representing how long addresses are
  word_size: int representing how long a word on the memory matrix is
  """
  def __init__(self, address_length, _word_size):
    self.defaults = {
      'allocation_gate': 1,
      'write_gate': 1,
      'read_mode': 3,
      'read_address': address_length,
      'write_address': address_length,
    }
    self.keys = sorted(self.defaults.keys())

  def create_content(self, content, mu):
    return ROMContent(content, mu, self.defaults)

  # batch x rom_size
  def read_rom_batch_tensor(self, batch_tensor):
    shape = tf.shape(batch_tensor)
    mu_tensor = batch_tensor[:, 0]
    result = {}

    index = 1
    for key in self.keys:
      result[key] = batch_tensor[:, index:index+1+self.defaults[key]] # Plus 1 because we need the presence of it as well
      index += 1 + self.defaults[key]

    return result

  def read_single_tensor(self, key, tensor, index):
    if tensor[index + self.defaults[key]] == 1:
      return tensor[index:index+self.defaults[key]]

  def read_rom_array(self, array):
    result = {}
    mu =  array[0]

    array = array[1:]
    for key in self.keys:
      if array[0] == 1:
        result[key] = array[0:self.defaults[key]]
      array = array[self.defaults[key]+1:]  # Plus 1 because the next bit determines if the vector is present

    return ROMContent(result, mu, self.defaults)

class ROMContent:
  def __init__(self, content, mu, defaults):
    self.content = {
      'mu': mu
    }
    self.defaults = defaults
    for key, default_length in defaults.items():
      if key in content:
        self.content[key] = content[key]

  def to_array(self):
    result = [self.content['mu']]
    keys = sorted(self.defaults.keys())
    for key in keys:
      result += [1] if key in self.content else [0]
      result += self.content[key] if key in self.content else [0] * self.defaults[key]

    return result
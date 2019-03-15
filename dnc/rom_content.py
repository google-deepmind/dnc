import tensorflow as tf

class ROMContentFactory:

  """
  content: dict of values to be put on the rom
  address_length: int representing how long addresses are
  word_size: int representing how long a word on the memory matrix is
  """
  def __init__(self, key_size, memory_size, _word_size):
    self.defaults = {
      'allocation_gate': 1,
      'write_gate': 1,
      'read_mode': 3,
      'read_weight': memory_size,
      'write_weight': memory_size,
    }
    self.key_size = key_size
    self.keys = sorted(self.defaults.keys())

  # Idea for later: just add a boolean for when there must be a key on this part. Then a new key is generated.
  # Now, hardcoding seems fine
  """
  key: The key that is matched by the output of the controller. Typically, only starts of programs have a specific key.
       Default key is zeros. Key must be of length key_size
  content: The content that is put on the rom
  mu: The value of mu for this content     
  """
  def create_content(self, key, content, mu):
    return ROMContent(key, content, mu, self.defaults)

  # batch x rom_size
  # key is not output here, because
  def read_rom_batch_tensor(self, batch_tensor):
    # shape = tf.shape(batch_tensor)
    mu_tensor = batch_tensor[:, self.key_size:self.key_size+1]
    result = {'mu': mu_tensor}

    index = self.key_size + 1
    for key in self.keys:
      result[key] = batch_tensor[:, index:index+1+self.defaults[key]] # Plus 1 because we need the presence of it as well
      index += 1 + self.defaults[key]

    return result


class ROMContent:
  def __init__(self, key, content, mu, defaults):
    self.content = {
      'mu': mu,
      'key': key
    }
    self.defaults = defaults
    for key, default_length in defaults.items():
      if key in content:
        self.content[key] = content[key]

  def to_array(self):
    result = self.content['key'] + [self.content['mu']]
    keys = sorted(self.defaults.keys())
    for key in keys:
      result += [1] if key in self.content else [0]
      result += self.content[key] if key in self.content else [0] * self.defaults[key]

    return result

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

  def create_content(self, content, mu):
    return ROMContent(content, mu, self.defaults)

  def read_rom_array(self, array):
    result = {}
    mu =  array[0]

    array = array[1:]
    keys = sorted(self.defaults.keys())
    for key in keys:
      if array[self.defaults[key]] == 1:
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
      result += self.content[key] if key in self.content else [0] * self.defaults[key]
      result += [1] if key in self.content else [0]

    return result
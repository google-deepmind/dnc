import tensorflow as tf

from dnc import rom
# import rom

class RomTest(tf.test.TestCase):

  def testModule(self):
    content = tf.constant([[1,2,3,4],[1,3,5,7],[9,8,7,6]], dtype='float32')
    module = rom.ROM(content)

    read_key = tf.constant([[1,2,3,4],[0,3,5,7], [1,1,1,1]], dtype='float32')
    read_strength = tf.constant([[10000], [10000], [1]], dtype='float32')
    read_mode = tf.constant([[1,0], [1,0], [0,1]], dtype='float32')
    prev_read_weight = tf.constant([[0,0.5,0.5], [0.3,0.4,0.3], [0,1,0]], dtype='float32')

    read_word, read_weight = module(read_key, read_strength, read_mode, prev_read_weight)

    with self.test_session():
      read_word_np, read_weight_np = read_word.eval(), read_weight.eval()

    self.assertAllClose(
      read_word_np,
      [[1,2,3,4], [1,3,5,7], [9,8,7,6]],
    )

    self.assertAllClose(
      read_weight_np,
      [[1,0,0], [0,1,0], [0,0,1]]
    )

class MixerTest(tf.test.TestCase):

  def testModule(self):
    module = rom.Mixer()

    address_controller = tf.constant([[0, 1, 0], [0, 1, 0]], dtype='float32')
    address_rom = tf.constant([[0, 0, 1], [0, 0, 1]], dtype='float32')

    mu_controller = tf.constant([[1], [0]], dtype='float32')
    mu_rom = tf.constant([[1], [1]], dtype='float32')
    previous_mu = tf.constant([[1], [0.5]], dtype='float32')

    address, mu = module(address_controller, address_rom, mu_controller, mu_rom, previous_mu)

    print(address)

    with self.test_session():
      address_np, mu_np = address.eval(), mu.eval()

    self.assertAllClose(
      address_np,
      [[0, 0, 1],
      [0, 0.5, 0.5]]
    )

    self.assertAllClose(
      mu,
      [1, 0.5]
    )

if __name__ == '__main__':
  tf.test.main()

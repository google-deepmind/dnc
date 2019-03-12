import tensorflow as tf

from dnc import rom_content

class RomContentTest(tf.test.TestCase):
  def testModule(self):
    fac = rom_content.ROMContentFactory(2, 3)

    content1 = fac.create_content({'allocation_gate': [2], 'read_mode': [1,2,3]}, 0.5)
    content2 = fac.create_content({'write_gate': [0.3], 'read_address': [0.6, 0.6]}, 0.9)

    batch = [content1.to_array(), content2.to_array()]
    batch_tensor = tf.constant(batch, dtype=tf.float32)

    batch_reading = fac.read_rom_batch_tensor(batch_tensor)

    # Do .eval() here
    with self.test_session() as sess:
      print(sess.run(batch_reading))

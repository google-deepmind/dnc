# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example script to train the DNC on a repeated copy task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt

from dnc import dnc
from dnc import repeat_copy
from tasks import repeat_sequence

FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("hidden_size", 64, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size", 16, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 16, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 1, "Number of memory read heads.")
tf.flags.DEFINE_integer("clip_value", 20,
                        "Maximum absolute value of controller and dnc outputs.")

# Optimizer parameters.
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10,
                      "Epsilon used for RMSProp optimizer.")

# Task parameters
tf.flags.DEFINE_integer("batch_size", 16, "Batch size for training.")
tf.flags.DEFINE_integer("num_bits", 4, "Dimensionality of each vector to copy")
tf.flags.DEFINE_integer(
    "min_length", 1,
    "Lower limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer(
    "max_length", 2,
    "Upper limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer("min_repeats", 1,
                        "Lower limit on number of copy repeats.")
tf.flags.DEFINE_integer("max_repeats", 2,
                        "Upper limit on number of copy repeats.")

# Training options.
tf.flags.DEFINE_integer("num_training_iterations", 100000,
                        "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 100,
                        "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_string("checkpoint_dir", "./checkpoint",
                       "Checkpointing directory.")
tf.flags.DEFINE_integer("checkpoint_interval", 1000,
                        "Checkpointing step interval.")
tf.flags.DEFINE_string("summary_dir", "./summaries", "Summary directoyu")

def run_model(input_sequence, output_size, return_weights, time_major=False):
  """Runs model on input sequence."""

  access_config = {
      "memory_size": FLAGS.memory_size,
      "word_size": FLAGS.word_size,
      "num_reads": FLAGS.num_read_heads,
      "num_writes": FLAGS.num_write_heads,
  }
  controller_config = {
      "hidden_size": FLAGS.hidden_size,
  }
  clip_value = FLAGS.clip_value

  dnc_core = dnc.DNC(access_config, controller_config, output_size, clip_value, return_weights=return_weights)
  initial_state = dnc_core.initial_state(FLAGS.batch_size)
  output_sequence, _ = tf.nn.dynamic_rnn(
      cell=dnc_core,
      inputs=input_sequence,
      time_major=time_major,
      initial_state=initial_state
  )

  return output_sequence


def train(num_training_iterations, report_interval):
  """Trains the DNC and periodically reports the loss."""

  dataset = repeat_copy.RepeatCopy(FLAGS.num_bits, FLAGS.batch_size,
                                   FLAGS.min_length, FLAGS.max_length,
                                   FLAGS.min_repeats, FLAGS.max_repeats)
  dataset_tensors = dataset()

  output_concat = run_model(dataset_tensors.observations, dataset.target_size, True, time_major=dataset.time_major())

  output_logits = output_concat[:, :, 0:dataset.target_size]
  output_read_weightings = output_concat[:, :, dataset.target_size:(dataset.target_size+FLAGS.memory_size)]
  output_write_weightings = output_concat[:, :, (dataset.target_size+FLAGS.memory_size):(dataset.target_size+2*FLAGS.memory_size)]

  # Used for visualization.
  # output = tf.round(
  #     tf.expand_dims(dataset_tensors.mask, -1) * tf.sigmoid(output_logits))

  output = tf.round(tf.sigmoid(output_logits))

  print('Target: ')
  print(dataset_tensors.target)
  print('\nLogits: ')
  print(output_logits)

  train_loss = dataset.cost(output_logits, dataset_tensors.target)

  tf.summary.image('Input', tf.expand_dims(dataset_tensors.observations, 3))
  tf.summary.image('Target', tf.expand_dims(dataset_tensors.target, 3))
  tf.summary.image('Output', tf.expand_dims(output_logits, 3))
  tf.summary.image('Read_weightings', tf.expand_dims(output_read_weightings, 3))
  tf.summary.image('Write_weightings', tf.expand_dims(output_write_weightings, 3))
  tf.summary.histogram('Loss', train_loss)

  merged = tf.summary.merge_all()

  # Set up optimizer with global norm clipping.
  trainable_variables = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(
      tf.gradients(train_loss, trainable_variables), FLAGS.max_grad_norm)

  global_step = tf.get_variable(
      name="global_step",
      shape=[],
      dtype=tf.int64,
      initializer=tf.zeros_initializer(),
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

  optimizer = tf.train.RMSPropOptimizer(
      FLAGS.learning_rate, epsilon=FLAGS.optimizer_epsilon)
  train_step = optimizer.apply_gradients(
      zip(grads, trainable_variables), global_step=global_step)

  saver = tf.train.Saver()

  if FLAGS.checkpoint_interval > 0:
    hooks = [
        tf.train.CheckpointSaverHook(
            checkpoint_dir=FLAGS.checkpoint_dir,
            save_steps=FLAGS.checkpoint_interval,
            saver=saver)
    ]
  else:
    hooks = []

  # Train.
  with tf.train.SingularMonitoredSession(
      hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir) as sess:

    train_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

    start_iteration = sess.run(global_step)
    total_loss = 0

    for train_iteration in range(start_iteration, num_training_iterations):
      _, loss, summary = sess.run([train_step, train_loss, merged])
      total_loss += loss

      if (train_iteration + 1) % report_interval == 0:
        dataset_tensors_np, output_np = sess.run([dataset_tensors, output])
        dataset_string = dataset.to_human_readable(dataset_tensors_np,
                                                   output_np)

        tf.logging.info("%d: Avg training loss %f.\n%s",
                        train_iteration, total_loss / report_interval,
                        dataset_string)
        total_loss = 0

        train_writer.add_summary(summary, train_iteration)

# def run_saved():

def to_batch_major(tensor):
  return tf.transpose(tensor, [1, 0, 2])

def main(unused_argv):
  tf.logging.set_verbosity(3)  # Print INFO log messages.
  train(FLAGS.num_training_iterations, FLAGS.report_interval)


if __name__ == "__main__":
  tf.app.run()

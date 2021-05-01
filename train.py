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

import tensorflow.compat.v1 as tf1
import tensorflow as tf
import sonnet as snt
import datetime

from dnc import dnc
from dnc import repeat_copy

FLAGS = tf1.flags.FLAGS

# Model parameters
tf1.flags.DEFINE_integer("hidden_size", 64, "Size of LSTM hidden layer.")
tf1.flags.DEFINE_integer("memory_size", 16, "The number of memory slots.")
tf1.flags.DEFINE_integer("word_size", 16, "The width of each memory slot.")
tf1.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf1.flags.DEFINE_integer("num_read_heads", 4, "Number of memory read heads.")
tf1.flags.DEFINE_integer("clip_value", 20,
                        "Maximum absolute value of controller and dnc outputs.")

# Optimizer parameters.
tf1.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf1.flags.DEFINE_float("learning_rate", 1e-4, "Optimizer learning rate.")
tf1.flags.DEFINE_float("optimizer_epsilon", 1e-10,
                      "Epsilon used for RMSProp optimizer.")

# Task parameters
tf1.flags.DEFINE_integer("batch_size", 16, "Batch size for training.")
tf1.flags.DEFINE_integer("num_bits", 4, "Dimensionality of each vector to copy")
tf1.flags.DEFINE_integer(
    "min_length", 1,
    "Lower limit on number of vectors in the observation pattern to copy")
tf1.flags.DEFINE_integer(
    "max_length", 2,
    "Upper limit on number of vectors in the observation pattern to copy")
tf1.flags.DEFINE_integer("min_repeats", 1,
                        "Lower limit on number of copy repeats.")
tf1.flags.DEFINE_integer("max_repeats", 2,
                        "Upper limit on number of copy repeats.")

# Training options.
tf1.flags.DEFINE_integer("num_training_iterations", 10000,
                        "Number of iterations to train for.")
tf1.flags.DEFINE_integer("report_interval", 100,
                        "Iterations between reports (samples, valid loss).")
tf1.flags.DEFINE_string("checkpoint_dir", "./logs/dnc/checkpoint",
                       "Checkpointing directory.")
tf1.flags.DEFINE_integer("checkpoint_interval", 2000,
                        "Checkpointing step interval.")

@tf.function
def train_step(x, y, rnn_model, loss, optimizer):
  """Runs model on input sequence."""
  initial_state = rnn_model.get_initial_state()
  with tf.GradientTape() as tape:
    output_sequence, _ = tf.compat.v1.nn.dynamic_rnn(
        cell=rnn_model,
        inputs=x,
        time_major=True,
        initial_state=initial_state)
    loss_value = loss(output_sequence, y)
  grads = tape.gradient(loss_value, rnn_model.trainable_variables)
  grads, _ = tf.clip_by_global_norm(grads, FLAGS.max_grad_norm)
  optimizer.apply_gradients(zip(grads, rnn_model.trainable_variables))

  return loss_value

@tf.function
def test_step(x, y, rnn_model, loss, mask):
  initial_state = rnn_model.get_initial_state()
  output_sequence, _ = tf.compat.v1.nn.dynamic_rnn(
    cell=rnn_model,
    inputs=x,
    time_major=True,
    initial_state=initial_state)
  loss_value = loss(output_sequence, y)
  # Used for visualization.
  output = tf.round(
    tf.expand_dims(mask, -1) * tf.sigmoid(output_sequence))
  return loss_value, output


def train(num_training_iterations, report_interval):
  """Trains the DNC and periodically reports the loss."""

  dataset = repeat_copy.RepeatCopy(FLAGS.num_bits, FLAGS.batch_size,
                                   FLAGS.min_length, FLAGS.max_length,
                                   FLAGS.min_repeats, FLAGS.max_repeats,
                                   dtype=tf.float64)
  dataset_tensors = dataset()

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

  dnc_core = dnc.DNC(
    access_config, controller_config, dataset.target_size, FLAGS.batch_size, clip_value)
  loss_fn = lambda pred, target: dataset.cost(
    pred, target, dataset_tensors.mask)
  optimizer = tf.compat.v1.train.RMSPropOptimizer(
      FLAGS.learning_rate, epsilon=FLAGS.optimizer_epsilon)

  #saver = tf.train.Checkpoint()

  # Set up logging and metrics
  train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
  test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  train_log_dir = 'logs/dnc/' + current_time + '/train'
  test_log_dir = 'logs/dnc/' + current_time + '/test'
  train_summary_writer = tf.summary.create_file_writer(train_log_dir)
  test_summary_writer = tf.summary.create_file_writer(test_log_dir)

  # Test once to initialize
  graph_log_dir = 'logs/dnc/' + current_time + '/graph'
  graph_writer = tf.summary.create_file_writer(graph_log_dir)
  with graph_writer.as_default():
    tf.summary.trace_on(graph=True, profiler=True)
    test_step(
      dataset_tensors.observations, dataset_tensors.target, dnc_core, loss_fn, dataset_tensors.mask
    )
    tf.summary.trace_export(
      name="dnc_trace",
      step=0,
      profiler_outdir=graph_log_dir)
  return

  # Set up model checkpointing
  checkpoint = tf.train.Checkpoint(model=dnc_core, optimizer=optimizer)

  # Train.
  for epoch in range(0, num_training_iterations):
    loss_value = train_step(
      dataset_tensors.observations, dataset_tensors.target, dnc_core, loss_fn, optimizer,
    )
    train_loss(loss_value)
    with train_summary_writer.as_default():
      tf.summary.scalar('loss', train_loss.result(), step=epoch)

    if (epoch) % report_interval == 0:
      loss_value, output = test_step(
        dataset_tensors.observations, dataset_tensors.target, dnc_core, loss_fn, dataset_tensors.mask
      )
      test_loss(loss_value)
      #dataset_string = dataset.to_human_readable(dataset_tensors_np,output_np)
      with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)

      template = 'Epoch {}, Loss: {}, Test Loss: {}'
      print(template.format(
        epoch + 1,
        train_loss.result(),
        test_loss.result(),
      ))

      # reset metrics every epoch
      train_loss.reset_states()
      test_loss.reset_states()

    if (epoch) % FLAGS.checkpoint_interval == 0:
      checkpoint.save(FLAGS.checkpoint_dir)


def main(unused_argv):
  tf.compat.v1.logging.set_verbosity(3)  # Print INFO log messages.
  train(FLAGS.num_training_iterations, FLAGS.report_interval)


if __name__ == "__main__":
  tf.compat.v1.app.run()

from __future__ import print_function

import copy4_data
from dnc import dnc_feedforward
import tensorflow as tf

import sonnet as snt

import numpy as np
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

from copy4_data import Copy4Data

from dnc import access

BATCH_SIZE = 8
CHECKPOINT_INTERVAL = 1000

MAX_GRAD_NORM = 50
LEARNING_RATE = 1e-4
OPTIMIZER_EPSILON = 1e-10

NUM_TRAINING_ITERATIONS = 100000
REPORT_INTERVAL = 100

SUMMARY_DIR = '/summaries'

def run_model(input_sequence, return_weights=False):

  # copied from train.py
  access_config = {
      "memory_size": 8,
      "word_size": 8,
      "num_reads": 1,
      "num_writes": 1,
  }
  controller_config = {
      "hidden_size": 64,
  }
  clip_value = 20

  dnc_core = dnc_feedforward.DNCfeedforward(access_config, controller_config, 10, clip_value, return_weights=return_weights)
  initial_state = dnc_core.initial_state(BATCH_SIZE)

  output, _ = tf.nn.dynamic_rnn(
    cell=dnc_core,
    inputs=input_sequence,
    time_major=False,
    initial_state=initial_state
  )

  return output

def train(num_training_iterations, report_interval):
  dataset = Copy4Data(10, BATCH_SIZE)

  dataset_tensors = dataset()

  output_concat = run_model(dataset_tensors.observations, return_weights=True)
  output_logits = output_concat[:, :, 0:10]
  output_read_weightings = output_concat[:, :, 10:18]
  output_write_weightings = output_concat[:, :, 18:26]

  # Used for visualization.
  output = tf.round(tf.sigmoid(output_logits))

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
    tf.gradients(train_loss, trainable_variables), MAX_GRAD_NORM)

  global_step = tf.get_variable(
    name="global_step",
    shape=[],
    dtype=tf.int64,
    initializer=tf.zeros_initializer(),
    trainable=False,
    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

  optimizer = tf.train.RMSPropOptimizer(
    LEARNING_RATE, epsilon=OPTIMIZER_EPSILON)
  train_step = optimizer.apply_gradients(
    zip(grads, trainable_variables), global_step=global_step)

  saver = tf.train.Saver()

  if CHECKPOINT_INTERVAL > 0:
    hooks = [
      tf.train.CheckpointSaverHook(
        checkpoint_dir='./checkpoint',
        save_steps=CHECKPOINT_INTERVAL,
        saver=saver)
    ]
  else:
    hooks = []

  with tf.train.SingularMonitoredSession(
      hooks=hooks, checkpoint_dir='./checkpoint') as sess:

    tf.summary.FileWriter('./summaries', sess.graph)

    start_iteration = sess.run(global_step)

    total_loss = 0

    train_writer = tf.summary.FileWriter('./summaries', sess.graph)

    for train_iteration in range(start_iteration, num_training_iterations):
      _, loss = sess.run([train_step, train_loss])
      total_loss += loss

      if (train_iteration + 1) % report_interval == 0:
        dataset_tensors_np, output_np, summary = sess.run([dataset_tensors, output, merged])

        dataset_string = dataset.to_human_readable(dataset_tensors_np,
                                                   output_np)
        tf.logging.info("%d: Avg training loss %f.\n%s",
                        train_iteration, total_loss / report_interval,
                        dataset_string)

        train_writer.add_summary(summary, train_iteration)

        total_loss = 0

def run_saved():
  dataset = Copy4Data(10, BATCH_SIZE)
  dataset_tensors = dataset()

  output_concat = run_model(dataset_tensors.observations, return_weights=True)
  output_logits = output_concat[:, :, 0:10]
  output_read_weightings = output_concat[:, :, 10:18]
  output_write_weightings = output_concat[:, :, 18:26]

  # Used for visualization.
  output = tf.round(tf.sigmoid(output_logits))

  train_loss = dataset.cost(output_logits, dataset_tensors.target)

  tf.summary.image('Input', tf.expand_dims(to_batch_major(dataset_tensors.observations), 3))
  tf.summary.image('Target', tf.expand_dims(to_batch_major(dataset_tensors.target), 3))
  tf.summary.image('Output', tf.expand_dims(to_batch_major(output_logits), 3))
  tf.summary.image('Read_weightings', tf.expand_dims(to_batch_major(output_read_weightings), 3))
  tf.summary.image('Write_weightings', tf.expand_dims(to_batch_major(output_write_weightings), 3))

  saver = tf.train.Saver()

  with tf.Session() as sess:
    saver.restore(sess, "./checkpoint/model.ckpt-34000")
    print("Model restored.")

    dataset_tensors_np, output_np, output_weightings_read_np, output_weightings_write_np, loss = \
      sess.run([dataset_tensors, output, output_weightings_read, output_weightings_write, train_loss])
    dataset_string = dataset.to_human_readable(dataset_tensors_np,
                                               output_np)

    tf.logging.info("Train loss of batch: %d.\n%s", loss, dataset_string)

def to_batch_major(tensor):
  return tf.transpose(tensor, [1, 0, 2])

def main(unused_argv):
  tf.logging.set_verbosity(3)  # Print INFO log messages.
  train(NUM_TRAINING_ITERATIONS, REPORT_INTERVAL)
  # run_saved()

if __name__ == "__main__":
  tf.app.run()

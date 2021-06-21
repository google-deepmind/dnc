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

import argparse
import datetime
import tensorflow as tf

from dnc import dnc
from dnc import repeat_copy

parser = argparse.ArgumentParser(description="Train DNC for repeat copy task.")

# Model parameters
parser.add_argument(
    "--hidden_size", default=64, type=int, help="Size of LSTM hidden layer."
)
parser.add_argument(
    "--memory_size", default=16, type=int, help="The number of memory slots."
)
parser.add_argument(
    "--word_size", default=16, type=int, help="The width of each memory slot."
)
parser.add_argument(
    "--num_write_heads", default=1, type=int, help="Number of memory write heads."
)
parser.add_argument(
    "--num_read_heads", default=4, type=int, help="Number of memory read heads."
)
parser.add_argument(
    "--clip_value",
    default=20,
    type=int,
    help="Maximum absolute value of controller and dnc outputs.",
)

# Optimizer parameters.
parser.add_argument(
    "--max_grad_norm", default=50, type=float, help="Gradient clipping norm limit."
)
parser.add_argument(
    "--learning_rate", default=1e-4, type=float, help="Optimizer learning rate."
)
parser.add_argument(
    "--optimizer_epsilon",
    default=1e-10,
    type=float,
    help="Epsilon used for RMSProp optimizer.",
)

# Task parameters
parser.add_argument(
    "--batch_size", default=16, type=int, help="Batch size for training."
)
parser.add_argument(
    "--num_bits", default=4, type=int, help="Dimensionality of each vector to copy"
)
parser.add_argument(
    "--min_length",
    default=2,
    type=int,
    help="Lower limit on number of vectors in the observation pattern to copy",
)
parser.add_argument(
    "--max_length",
    default=3,
    type=int,
    help="Upper limit on number of vectors in the observation pattern to copy",
)
parser.add_argument(
    "--min_repeats", default=1, type=int, help="Lower limit on number of copy repeats."
)
parser.add_argument(
    "--max_repeats", default=3, type=int, help="Upper limit on number of copy repeats."
)

# Training options.
parser.add_argument(
    "--epochs", default=10000, type=int, help="Number of epochs to train for."
)
parser.add_argument(
    "--log_dir", default="./logs/repeat_copy", type=str, help="Logging directory."
)
parser.add_argument(
    "--report_interval",
    default=500,
    type=int,
    help="Epochs between reports (samples, valid loss).",
)
parser.add_argument(
    "--checkpoint_interval", default=2000, type=int, help="Checkpointing step interval."
)

FLAGS = parser.parse_args()


def train_step(dataset_tensors, rnn_model, optimizer, loss_fn):
    return train_step_graphed(
        dataset_tensors.observations,
        dataset_tensors.target,
        dataset_tensors.mask,
        rnn_model,
        optimizer,
        loss_fn,
    )


@tf.function
def train_step_graphed(
    x,
    y,
    mask,
    rnn_model,
    optimizer,
    loss_fn,
):
    """Runs model on input sequence."""
    initial_state = rnn_model.get_initial_state(x)
    with tf.GradientTape() as tape:
        output_sequence = rnn_model(
            inputs=x,
            initial_state=initial_state,
        )
        loss_value = loss_fn(output_sequence, y, mask)
    grads = tape.gradient(loss_value, rnn_model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, FLAGS.max_grad_norm)
    optimizer.apply_gradients(zip(grads, rnn_model.trainable_variables))
    return loss_value


def test_step(dataset_tensors, rnn_model, optimizer, loss_fn):
    return test_step_graphed(
        dataset_tensors.observations,
        dataset_tensors.target,
        dataset_tensors.mask,
        rnn_model,
        loss_fn,
    )


@tf.function
def test_step_graphed(
    x,
    y,
    mask,
    rnn_model,
    loss_fn,
):
    initial_state = rnn_model.get_initial_state(x)
    output_sequence = rnn_model(
        inputs=x,
        initial_state=initial_state,
    )
    loss_value = loss_fn(output_sequence, y, mask)
    # Used for visualization.
    output = tf.round(tf.expand_dims(mask, -1) * tf.sigmoid(output_sequence))
    return loss_value, output


def train(num_training_iterations, report_interval):
    """Trains the DNC and periodically reports the loss."""

    train_dataset = repeat_copy.RepeatCopy(
        FLAGS.num_bits,
        FLAGS.batch_size,
        FLAGS.min_length,
        FLAGS.max_length,
        FLAGS.min_repeats,
        FLAGS.max_repeats,
        dtype=tf.float32,
    )
    # Generate test data with double maximum repeat length
    test_dataset = repeat_copy.RepeatCopy(
        FLAGS.num_bits,
        100,  # FLAGS.batch_size,
        FLAGS.min_length,
        FLAGS.max_length,
        FLAGS.max_repeats * 2,
        FLAGS.max_repeats * 2,
        dtype=tf.float32,
    )

    dataset_tensor = train_dataset()
    test_dataset_tensor = test_dataset()

    access_config = {
        "memory_size": FLAGS.memory_size,
        "word_size": FLAGS.word_size,
        "num_reads": FLAGS.num_read_heads,
        "num_writes": FLAGS.num_write_heads,
    }
    controller_config = {
        # "hidden_size": FLAGS.hidden_size,
        "units": FLAGS.hidden_size,
    }
    clip_value = FLAGS.clip_value

    dnc_cell = dnc.DNC(
        access_config,
        controller_config,
        train_dataset.target_size,
        FLAGS.batch_size,
        clip_value,
    )
    dnc_core = tf.keras.layers.RNN(
        cell=dnc_cell,
        time_major=True,
        return_sequences=True,
    )
    optimizer = tf.compat.v1.train.RMSPropOptimizer(
        FLAGS.learning_rate, epsilon=FLAGS.optimizer_epsilon
    )
    loss_fn = train_dataset.cost

    # Set up logging and metrics
    train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean("test_loss", dtype=tf.float32)

    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = FLAGS.log_dir + "/train"
    test_log_dir = FLAGS.log_dir + "/test"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # Test once to initialize
    graph_log_dir = FLAGS.log_dir + "/graph"
    graph_writer = tf.summary.create_file_writer(graph_log_dir)
    with graph_writer.as_default():
        tf.summary.trace_on(graph=True, profiler=True)
        test_step(dataset_tensor, dnc_core, optimizer, loss_fn)
        tf.summary.trace_export(name="dnc_trace", step=0, profiler_outdir=graph_log_dir)

    # Set up model checkpointing
    checkpoint = tf.train.Checkpoint(model=dnc_core, optimizer=optimizer)
    manager = tf.train.CheckpointManager(
        checkpoint, FLAGS.log_dir + "/checkpoint", max_to_keep=10
    )

    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    # Train.
    for epoch in range(num_training_iterations):
        dataset_tensor = train_dataset()
        train_loss_value = train_step(dataset_tensor, dnc_core, optimizer, loss_fn)
        train_loss(train_loss_value)

        # report metrics
        if (epoch) % report_interval == 0:
            test_loss_value, output = test_step(
                test_dataset_tensor, dnc_core, optimizer, test_dataset.cost
            )
            test_loss(test_loss_value)
            with test_summary_writer.as_default():
                tf.summary.scalar("loss", test_loss.result(), step=epoch)
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", train_loss.result(), step=epoch)

            template = "Epoch {}, Loss: {}, Test Loss: {}"
            print(
                template.format(
                    epoch,
                    train_loss.result(),
                    test_loss.result(),
                )
            )

            dataset_string = test_dataset.to_human_readable(
                test_dataset_tensor, output.numpy()
            )
            print(dataset_string)

        # reset metrics every epoch
        train_loss.reset_states()
        test_loss.reset_states()

        # save model at defined intervals
        if (1 + epoch) % FLAGS.checkpoint_interval == 0:
            manager.save()
    # At the end, checkpoint as well
    manager.save()


def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(3)  # Print INFO log messages.
    train(FLAGS.epochs, FLAGS.report_interval)


if __name__ == "__main__":
    tf.compat.v1.app.run()

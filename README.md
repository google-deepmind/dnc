# Differentiable Neural Computer (DNC)

This package provides an implementation of the Differentiable Neural Computer,
as [published in Nature](
https://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz).

Any publication that discloses findings arising from using this source code must
cite “Hybrid computing using a neural network with dynamic external memory",
Nature 538, 471–476 (October 2016) doi:10.1038/nature20101.

## Introduction

The Differentiable Neural Computer is a recurrent neural network. At each
timestep, it has state consisting of the current memory contents (and auxiliary
information such as memory usage), and maps input at time `t` to output at time
`t`. It is implemented as a collection of `RNNCore` modules, which allow
plugging together the different modules to experiment with variations on the
architecture.

*   The *access* module is where the main DNC logic happens; as this is where
    memory is written to and read from. At every timestep, the input to an
    access module is a vector passed from the `controller`, and its output is
    the contents read from memory. It uses two futher `RNNCore`s:
    `TemporalLinkage` which tracks the order of memory writes, and `Freeness`
    which tracks which memory locations have been written to and not yet
    subsequently "freed". These are both defined in `addressing.py`.

*   The *controller* module "controls" memory access. Typically, it is just a
    feedforward or (possibly deep) LSTM network, whose inputs are the inputs to
    the overall recurrent network at that time, concatenated with the read
    memory output from the access module from the previous timestep.

*   The *dnc* simply wraps the access module and the control module, and forms
    the basic `RNNCore` unit of the overall architecture. This is defined in
    `dnc.py`.

![DNC architecture](images/dnc_model.png)

## Train
The `DNC` requires an installation of [TensorFlow](https://www.tensorflow.org/)
and [Sonnet](https://github.com/deepmind/sonnet). An example training script is
provided for the algorithmic task of repeatedly copying a given input string.
This can be executed from a python interpreter:

```shell
$ ipython train.py
```

You can specify training options, including parameters to the model
and optimizer, via flags:

```shell
$ python train.py --memory_size=64 --num_bits=8 --max_length=3

# Or with ipython:
$ ipython train.py -- --memory_size=64 --num_bits=8 --max_length=3
```

Periodically saving, or 'checkpointing', the model is disabled by default. To
enable, use the `checkpoint_interval` flag. E.g. `--checkpoint_interval=10000`
will ensure a checkpoint is created every `10,000` steps. The model will be
checkpointed to `/tmp/tf/dnc/` by default. From there training can be resumed.
To specify an alternate checkpoint directory, use the `checkpoint_dir` flag.
Note: ensure that `/tmp/tf/dnc/` is deleted before training is resumed with
different model parameters, to avoid shape inconsistency errors.

More generally, the `DNC` class found within `dnc.py` can be used as a standard
TensorFlow rnn core and unrolled with TensorFlow rnn ops, such as
`tf.nn.dynamic_rnn` on any sequential task.

Disclaimer: This is not an official Google product

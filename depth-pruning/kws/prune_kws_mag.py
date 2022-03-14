# Lint as: python3
"""
Pruning of KWS model using magnitude pruning
"""

import os

from absl import app
import keras_model as models
import get_dataset as kws_data
import kws_util
import pathlib

import tensorflow as tf
assert tf.__version__.startswith('2')

import argparse
import pathlib
import aux_models as aux_models
from contextlib import redirect_stdout

import tensorflow_model_optimization as tfmot

num_classes = 12 # should probably draw this directly from the dataset.
# FLAGS = None

def prune():

    # Base Model Setup
    base_model = tf.keras.models.load_model(FLAGS.base_model_path)

    # Data Setup
    print('We will download data to {:}'.format(FLAGS.data_dir))
    print('We will train for {:} epochs'.format(FLAGS.epochs))

    ds_train, ds_test, ds_val = kws_data.get_training_data(FLAGS)
    print("Done getting data")

    # this is taken from the dataset web page.
    # there should be a better way than hard-coding this
    train_shuffle_buffer_size = 85511
    val_shuffle_buffer_size = 10102
    test_shuffle_buffer_size = 4890

    ds_train = ds_train.shuffle(train_shuffle_buffer_size)
    ds_val = ds_val.shuffle(val_shuffle_buffer_size)
    ds_test = ds_test.shuffle(test_shuffle_buffer_size)

    # Define model for pruning.
    n_epochs = FLAGS.epochs
    n_epochs_recover = FLAGS.epochs_recover
    N_BATCHES = 856 #Calculated manually based on dataset
    pruning_end_step = n_epochs * N_BATCHES

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    if FLAGS.pruning_scheme=='polynomial':
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=FLAGS.start_sparsity,
                                                                    final_sparsity=FLAGS.end_sparsity,
                                                                    begin_step=0,
                                                                    end_step=pruning_end_step)
        }
    elif FLAGS.pruning_scheme=='constant':
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                                                                target_sparsity=FLAGS.end_sparsity,
                                                                begin_step=0,
                                                                end_step=pruning_end_step)
        }

    model_for_pruning = prune_low_magnitude(base_model, **pruning_params)

    #Save Model Info
    train_dir = pathlib.Path(FLAGS.train_dir + "/" + FLAGS.ckpt_name)
    train_dir.mkdir(parents=True, exist_ok=True)

    with open(FLAGS.train_dir + "/" + FLAGS.ckpt_name + "/" + FLAGS.ckpt_name + '_modelinfo.log', 'w') as f:
        with redirect_stdout(f):
            base_model.summary()
            model_for_pruning.summary()

    model_for_pruning.compile(
        #optimizer=keras.optimizers.RMSprop(learning_rate=args.learning_rate),  # Optimizer
        optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),  # Optimizer
        # Loss function to minimize
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    callbacks = kws_util.get_pruning_callbacks(args=FLAGS)
    train_hist = model_for_pruning.fit(ds_train, validation_data=ds_val, epochs=FLAGS.epochs, callbacks=callbacks)


if __name__ == '__main__':
    FLAGS, unparsed = kws_util.parse_command()

    prune()

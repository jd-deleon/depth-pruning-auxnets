# Lint as: python3
"""
Pruning of VWW model using magnitude pruning
"""

import os

from absl import app
from vww_model import mobilenet_v1

import tensorflow as tf
assert tf.__version__.startswith('2')

import argparse
import pathlib
import aux_models as aux_models
from contextlib import redirect_stdout

import tensorflow_model_optimization as tfmot

IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 20

def prune():
    # Base Model Setup
    base_model = tf.keras.models.load_model(FLAGS.base_model_path)

    # Data Setup
    validation_split = 0.1

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=.1,
        horizontal_flip=True,
        validation_split=validation_split,
        rescale=1. / 255)
    train_generator = datagen.flow_from_directory(
        FLAGS.data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='training',
        color_mode='rgb')
    val_generator = datagen.flow_from_directory(
        FLAGS.data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='validation',
        color_mode='rgb')
    print(train_generator.class_indices)

    # Define model for pruning.
    n_epochs = 50
    n_epochs_recover = FLAGS.epochs_recover
    N_BATCHES = 3084
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

    model = train_epochs(model_for_pruning, train_generator, val_generator, 20, 0.001, save_ckpt=True)
    model = train_epochs(model_for_pruning, train_generator, val_generator, 10, 0.0005, save_ckpt=True)
    model = train_epochs(model_for_pruning, train_generator, val_generator, 20, 0.00025, save_ckpt=True)


def train_epochs(model, train_generator, val_generator, epoch_count,
                 learning_rate, save_ckpt=False):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    train_dir = pathlib.Path(FLAGS.train_dir + "/" + FLAGS.ckpt_name)
    csv_logger = tf.keras.callbacks.CSVLogger(FLAGS.train_dir + "/" + FLAGS.ckpt_name + "/" + FLAGS.ckpt_name + '_' + str(learning_rate) + '.log')

    if save_ckpt==True:
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=(train_dir / (FLAGS.ckpt_name + "_{val_accuracy:.4f}")),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        history_fine = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epoch_count,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            batch_size=BATCH_SIZE,
            callbacks=[csv_logger, model_checkpoint_callback, tfmot.sparsity.keras.UpdatePruningStep()])
    else:
        history_fine = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epoch_count,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            batch_size=BATCH_SIZE,
            callbacks=[csv_logger, tfmot.sparsity.keras.UpdatePruningStep(),])
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--attach_tensor_name',
        help='Name of a tensor in the base model where the auxiliary network will be attached',
        type=str)
    parser.add_argument(
        '--aux_arch',
        help='Artchitecture for the auxiliary network (dscnn32_16, dscnn16_16_16, dscnn32, conv, dense)',
        type=str)
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Visual Wakewords dataset location (processed by tinyMLperf)')
    parser.add_argument(
        '--train_dir',
        type=str,
        default='aux_models/test_models',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--ckpt_name',
        type=str)
    parser.add_argument(
        '--nopretrained',
        help='Set to true to use uninitialized weights for the base model (architecture is matched)',
        type=str,
        default="False")
    parser.add_argument(
        '--base_model_path',
        help='Path of the base model in TF SavedModel format',
        type=str,
        default="False")
    parser.add_argument(
        '--frozen_head',
        help='Freeze the head (base model layers) during training of auxiliary network',
        type=str,
        default="True")
    parser.add_argument(
        '--pruning_scheme',
        help='Pruning scheme to use (constant or polynomial)',
        type=str)
    parser.add_argument(
        '--epochs_recover',
        help='Additional epochs to train after reaching end sparsity, used only for polynomial training scheme',
        default=0,
        type=int)
    parser.add_argument(
        '--start_sparsity',
        help='Starting sparsity used in polynomial training scheme',
        default=0.0,
        type=float)
    parser.add_argument(
        '--end_sparsity',
        help='Final sparsity of trained model',
        type=float)


    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    prune()

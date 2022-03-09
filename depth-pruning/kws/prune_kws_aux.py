# Lint as: python3
"""
Pruning of KWS model using auxiliary networks
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

IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 20

num_classes = 12 # should probably draw this directly from the dataset.
# FLAGS = None

def prune():
    # Base Model Setup
    if FLAGS.nopretrained == "True":
        print("Not using pretrained MobileNet model")
        base_model = models.get_model(args=FLAGS)
    else:
        print("Using pretrained MobileNet model")
        base_model = tf.keras.models.load_model(FLAGS.base_model_path)

    if FLAGS.frozen_head == "True":
        print("Training with frozen head layers")
        base_model.trainable = False
    else:
        print("Training with unfrozen head layers")


    # Find attachment layer index
    attachment_tensor_index = -1
    for index, item in enumerate(base_model.layers):
        if item.name == FLAGS.attach_tensor_name:
            attachment_tensor_index = index
            break
        else:
            index = -1

    # Define aux model with intermediate output
    num_classes=12
    if FLAGS.aux_arch == "dscnn32_16":
        # aux_model = aux_models.dwconv_aux_model(base_model.layers[attachment_tensor_index].output, [2, 32, 10, 4, 2, 2, 16, 3, 3, 1, 1])
        aux_model = aux_models.dscnn_aux_model(base_model.layers[attachment_tensor_index].output, num_classes, [2, 32, 3, 3, 1, 1, 16, 3, 3, 1, 1])
    elif FLAGS.aux_arch == "dscnn16_16_16":
        aux_model = aux_models.dscnn_aux_model(base_model.layers[attachment_tensor_index].output, num_classes, [3, 16, 3, 3, 1, 1, 16, 3, 3, 1, 1, 16, 3, 3, 1, 1])
    elif FLAGS.aux_arch == "dscnn32":
        aux_model = aux_models.dscnn_aux_model(base_model.layers[attachment_tensor_index].output, num_classes, [2, 32, 3, 3, 1, 1])
    elif FLAGS.aux_arch == "conv":
        aux_model = aux_models.conv_aux_model(base_model.layers[attachment_tensor_index].output, num_classes, [1, 11, 3, 3])
    elif FLAGS.aux_arch == "dense":
        aux_model = aux_models.dense_aux_model(base_model.layers[attachment_tensor_index].output, num_classes, [2, 64, 32])
    else:
        print("arch not found")
        return

    aux_model = tf.keras.Model(inputs=base_model.inputs, outputs=aux_model)
    aux_model.summary()

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

    #Save Model Info
    train_dir = pathlib.Path(FLAGS.train_dir + "/" + FLAGS.ckpt_name)
    train_dir.mkdir(parents=True, exist_ok=True)

    with open(FLAGS.train_dir + "/" + FLAGS.ckpt_name + "/" + FLAGS.ckpt_name + '_modelinfo.log', 'w') as f:
        with redirect_stdout(f):
            base_model.summary()
            aux_model.summary()

    aux_model.compile(
        #optimizer=keras.optimizers.RMSprop(learning_rate=args.learning_rate),  # Optimizer
        optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),  # Optimizer
        # Loss function to minimize
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    callbacks = kws_util.get_callbacks(args=FLAGS)
    train_hist = aux_model.fit(ds_train, validation_data=ds_val, epochs=FLAGS.epochs, callbacks=callbacks)

if __name__ == '__main__':
    FLAGS, unparsed = kws_util.parse_command()

    prune()

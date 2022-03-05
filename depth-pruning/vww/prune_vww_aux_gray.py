# Lint as: python3
"""
Pruning of VWW model using auxiliary networks. Modified from the original VWW task to use grayscale inputs for hardware deployment
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

IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 20

def prune():
    # Base Model Setup
    if FLAGS.nopretrained == "True":
        if FLAGS.arch == "mobilenetV1":
            print("Using uninitialized MobileNetV1 model")
            base_model = mobilenet_v1_gray()
        elif FLAGS.arch == "alexnet":
            print("Using uninitialized Alexnet model")
            base_model = alexnet_model()
        elif FLAGS.arch == "mobilenetV2":
            print("Using uninitialized MobileNetV2 model")
            base_model = model = tf.keras.applications.mobilenet_v2.MobileNetV2(
                input_shape=(96,96,1), alpha=0.25, include_top=True, weights=None, classes=2)
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
    num_classes=2
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
    elif FLAGS.aux_arch == "dense1":
        aux_model = aux_models.dense_aux_model(base_model.layers[attachment_tensor_index].output, num_classes, [2, 64])
    else:
        print("arch not found")
        return

    aux_model = tf.keras.Model(inputs=base_model.inputs, outputs=aux_model)
    aux_model.summary()

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
        color_mode='grayscale')
    val_generator = datagen.flow_from_directory(
        FLAGS.data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='validation',
        color_mode='grayscale')
    print(train_generator.class_indices)

    #Save Model Info
    
    train_dir = pathlib.Path(FLAGS.train_dir + "/" + FLAGS.ckpt_name)
    train_dir.mkdir(parents=True, exist_ok=True)

    with open(FLAGS.train_dir + "/" + FLAGS.ckpt_name + "/" + FLAGS.ckpt_name + '_modelinfo.log', 'w') as f:
        with redirect_stdout(f):
            base_model.summary()
            aux_model.summary()

    model = train_epochs(aux_model, train_generator, val_generator, 20, 0.001, save_ckpt=True)
    model = train_epochs(aux_model, train_generator, val_generator, 10, 0.0005, save_ckpt=True)
    model = train_epochs(aux_model, train_generator, val_generator, 20, 0.00025, save_ckpt=True)

    # Save model
    # if len(argv) >= 3:
    #     model.save(argv[2])
    # else:
    #     model.save(FLAGS.train_dir + "/" + FLAGS.ckpt_name + "_trained")


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
            callbacks=[csv_logger, model_checkpoint_callback])
    else:
        history_fine = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epoch_count,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            batch_size=BATCH_SIZE,
            callbacks=[csv_logger])
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
        '--arch',
        help='Architecture of base model, to be used with --nopretrained',
        type=str,
        )

    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    prune()

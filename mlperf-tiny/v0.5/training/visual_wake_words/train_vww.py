# Lint as: python3
"""Training for the visual wakewords person detection model.

The visual wakewords person detection model is a core model for the TinyMLPerf
benchmark suite. This script provides source for how the reference model was
created and trained, and can be used as a starting point for open submissions
using re-training.
"""

import os

from absl import app
from vww_model import mobilenet_v1
from alexnet_model import alexnet_model

import tensorflow as tf
assert tf.__version__.startswith('2')

import argparse

IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 20

BASE_DIR = os.path.join(os.getcwd(), 'vw_coco2014_96')


def main(argv):
  if len(argv) >= 2:
    model = tf.keras.models.load_model(argv[1])
  else:
    # model = mobilenet_v1()
    if FLAGS.arch=="mobilenetV1":
      model = mobilenet_v1()
    elif FLAGS.arch=="alexnet":
      model = alexnet_model()
    elif FLAGS.arch=="mobilenetV2":
      model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=(96,96,3), alpha=0.25, include_top=True, weights=None, classes=2)
    elif FLAGS.arch=="resnet50":
      model = tf.keras.applications.resnet50.ResNet50(
        input_shape=(96,96,3), include_top=True, weights=None, classes=2)
    elif FLAGS.arch=="nasnetMobile":
      model = tf.keras.applications.nasnet.NASNetMobile(
        input_shape=(96,96,3), include_top=True, weights=None, classes=2)
    elif FLAGS.arch=="efficientNetB0":
      model = tf.keras.applications.efficientnet.EfficientNetB0(
        input_shape=(96,96,3), include_top=True, weights=None, classes=2)

  model.summary()

  batch_size = 50
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
      BASE_DIR,
      target_size=(IMAGE_SIZE, IMAGE_SIZE),
      batch_size=BATCH_SIZE,
      subset='training',
      color_mode='rgb')
  val_generator = datagen.flow_from_directory(
      BASE_DIR,
      target_size=(IMAGE_SIZE, IMAGE_SIZE),
      batch_size=BATCH_SIZE,
      subset='validation',
      color_mode='rgb')
  print(train_generator.class_indices)

  model = train_epochs(model, train_generator, val_generator, 20, 0.001)
  model = train_epochs(model, train_generator, val_generator, 10, 0.0005)
  model = train_epochs(model, train_generator, val_generator, 20, 0.00025)

  # Save model HDF5
  # if len(argv) >= 3:
  #   model.save(argv[2])
  # else:
  #   model.save('trained_models/vww_96.h5')
  model.save(FLAGS.train_dir)


def train_epochs(model, train_generator, val_generator, epoch_count,
                 learning_rate):
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate),
      loss='categorical_crossentropy',
      metrics=['accuracy'])
  history_fine = model.fit(
      train_generator,
      steps_per_epoch=len(train_generator),
      epochs=epoch_count,
      validation_data=val_generator,
      validation_steps=len(val_generator),
      batch_size=BATCH_SIZE)
  return model


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
          '--arch',
          type=str)
  parser.add_argument(
      '--train_dir',
      type=str,
      default='out_model',
      help='Directory to write event logs and checkpoint.')

  FLAGS, unparsed = parser.parse_known_args()  
  app.run(main)

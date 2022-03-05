# Lint as: python3
"""

Evaluates pruned VWW models

"""

import os

from absl import app
from vww_model import mobilenet_v1

import tensorflow as tf
assert tf.__version__.startswith('2')

import argparse

IMAGE_SIZE = 96
BATCH_SIZE = 1
EPOCHS = 20

FLAGS=None

# MODEL_DIR = os.path.join(os.getcwd(), 'vww_96_mobilenet')
# BASE_DIR = os.path.join(os.getcwd(), 'vw_coco2014_96')

def evaluate():

  model = tf.keras.models.load_model(FLAGS.ckpt_name)
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
  val_generator = datagen.flow_from_directory(
      FLAGS.data_dir,
      target_size=(IMAGE_SIZE, IMAGE_SIZE),
      batch_size=BATCH_SIZE,
      subset='validation',
      color_mode='rgb')

  result = model.evaluate(val_generator)
  print(result)

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
        '--data_dir',
        type=str,
        help='Visual Wakewords dataset location (processed by tinyMLperf)')
    parser.add_argument(
        '--ckpt_name',
        type=str,
        help='Model directory to evaluate')

    FLAGS, unparsed = parser.parse_known_args()        
    evaluate()

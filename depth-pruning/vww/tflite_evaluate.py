# Lint as: python3
"""

Evaluates pruned and quantized VWW tflite models

"""

import os
import numpy as np

from absl import app
from vww_model import mobilenet_v1

import tensorflow as tf
assert tf.__version__.startswith('2')

import argparse

IMAGE_SIZE = 96
BATCH_SIZE = 1
EPOCHS = 20

FLAGS=None

# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices):
  global test_images

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_image_indices),), dtype=int)
  for i, test_image_index in enumerate(test_image_indices):
    test_image = test_images[test_image_index]
    test_label = test_labels[test_image_index]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      test_image = test_image / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    predictions[i] = output.argmax()

  return predictions

def evaluate():

  batch_size = 50
  validation_split = 0.1

  datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rotation_range=10,
      width_shift_range=0.05,
      height_shift_range=0.05,
      zoom_range=.1,
      horizontal_flip=True,
      validation_split=validation_split,
      rescale=1. / 255,
      dtype=tf.int8)

  val_generator = datagen.flow_from_directory(
      FLAGS.data_dir,
      target_size=(IMAGE_SIZE, IMAGE_SIZE),
      batch_size=BATCH_SIZE,
      subset='validation',
      color_mode='grayscale')

  # Load the TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path=FLAGS.model_input)
  interpreter.allocate_tensors()

  # Get input and output tensors.
  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()

  # print(len(val_generator))

  correct = 0
  for test_data_idx in range(len(val_generator)):
    test_image = val_generator[test_data_idx][0]

    # Test the model on random input data.
    if input_details['dtype'] == np.int8:
      input_scale, input_zero_point = input_details["quantization"]
      test_image = test_image / input_scale + input_zero_point
    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    
    # print(test_image)
    interpreter.set_tensor(input_details["index"], test_image[0])
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if output_data.argmax() == val_generator[test_data_idx][1].argmax():
      correct = correct+1
    # print(test_data_idx)

  print("Accuracy: " + str(correct / len(val_generator)))

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
        '--model_input',
        type=str,
        help='Model directory to evaluate')

    FLAGS, unparsed = parser.parse_known_args()        
    evaluate()

"""
Converts pruned VWW checkpoints to tflite format and quantize to int8
"""

import os

from absl import app

import tensorflow as tf
assert tf.__version__.startswith('2')

import argparse
import pathlib

def quantize():
    # if len(argv) != 2:
    #     raise app.UsageError('Usage: convert_vww.py <model_to_convert.h5>')
    model = tf.keras.models.load_model(FLAGS.model_input)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    out_dir = pathlib.Path(FLAGS.quant_model)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tf.io.gfile.GFile(FLAGS.quant_model+'/'+FLAGS.quant_model+'_float.tflite', 'wb') as float_file:
        float_file.write(tflite_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_dataset_gen():
        dataset_dir = os.path.join(FLAGS.data_dir, "person")
        for idx, image_file in enumerate(os.listdir(dataset_dir)):
            # 10 representative images should be enough for calibration.
            if idx > 10:
                return
            full_path = os.path.join(dataset_dir, image_file)
            if os.path.isfile(full_path):
                img = tf.keras.preprocessing.image.load_img(
                    full_path, color_mode='grayscale').resize((96, 96))
                arr = tf.keras.preprocessing.image.img_to_array(img)
                # Scale input to [0, 1.0] like in training.
                yield [arr.reshape(1, 96, 96, 1) / 255.]

    # Convert model to full-int8 and save as quantized tflite flatbuffer.
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    quantized_tflite_model = converter.convert()
    with tf.io.gfile.GFile(FLAGS.quant_model+'/'+FLAGS.quant_model+'_int8.tflite', 'wb') as quantized_file:
        quantized_file.write(quantized_tflite_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Visual Wakewords dataset location (processed by tinyMLperf)')
    parser.add_argument(
        '--model_input',
        type=str,
        help='Model directory to quantize')
    parser.add_argument(
        '--quant_model',
        type=str,
        help='Output model directory')

    FLAGS, unparsed = parser.parse_known_args()   

    quantize()

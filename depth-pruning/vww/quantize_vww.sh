#!/bin/sh

DATA_DIR=$PWD"/../../mlperf-tiny/training/visual_wake_words/vw_coco2014_96"

export CUDA_VISIBLE_DEVICES=0

# Attachment Location
python tflite_convert_quantize.py --model_input="ckpt_vww_mobilenet-gray_aux_dscnn/dscnn_conv1" --data_dir=$DATA_DIR --quant_model="quant_vww_mobilenet_prunedC1" 
python tflite_convert_quantize.py --model_input="ckpt_vww_mobilenet-gray_aux_dscnn/dscnn_block1" --data_dir=$DATA_DIR --quant_model="quant_vww_mobilenet_prunedB1" 
python tflite_convert_quantize.py --model_input="ckpt_vww_mobilenet-gray_aux_dscnn/dscnn_block2" --data_dir=$DATA_DIR --quant_model="quant_vww_mobilenet_prunedB2" 
python tflite_convert_quantize.py --model_input="ckpt_vww_mobilenet-gray_aux_dscnn/dscnn_block3" --data_dir=$DATA_DIR --quant_model="quant_vww_mobilenet_prunedB3" 
python tflite_convert_quantize.py --model_input="ckpt_vww_mobilenet-gray_aux_dscnn/dscnn_block4" --data_dir=$DATA_DIR --quant_model="quant_vww_mobilenet_prunedB4" 
python tflite_convert_quantize.py --model_input="ckpt_vww_mobilenet-gray_aux_dscnn/dscnn_block5" --data_dir=$DATA_DIR --quant_model="quant_vww_mobilenet_prunedB5" 
python tflite_convert_quantize.py --model_input="ckpt_vww_mobilenet-gray_aux_dscnn/dscnn_block6" --data_dir=$DATA_DIR --quant_model="quant_vww_mobilenet_prunedB6" 
python tflite_convert_quantize.py --model_input="ckpt_vww_mobilenet-gray_aux_dscnn/dscnn_block7" --data_dir=$DATA_DIR --quant_model="quant_vww_mobilenet_prunedB7" 
python tflite_convert_quantize.py --model_input="ckpt_vww_mobilenet-gray_aux_dscnn/dscnn_block8" --data_dir=$DATA_DIR --quant_model="quant_vww_mobilenet_prunedB8" 
python tflite_convert_quantize.py --model_input="ckpt_vww_mobilenet-gray_aux_dscnn/dscnn_block9" --data_dir=$DATA_DIR --quant_model="quant_vww_mobilenet_prunedB9" 
python tflite_convert_quantize.py --model_input="ckpt_vww_mobilenet-gray_aux_dscnn/dscnn_block10" --data_dir=$DATA_DIR --quant_model="quant_vww_mobilenet_prunedB10" 
python tflite_convert_quantize.py --model_input="ckpt_vww_mobilenet-gray_aux_dscnn/dscnn_block11" --data_dir=$DATA_DIR --quant_model="quant_vww_mobilenet_prunedB11" 
python tflite_convert_quantize.py --model_input="ckpt_vww_mobilenet-gray_aux_dscnn/dscnn_block12" --data_dir=$DATA_DIR --quant_model="quant_vww_mobilenet_prunedB12" 
python tflite_convert_quantize.py --model_input="ckpt_vww_mobilenet-gray_aux_dscnn/dscnn_block13" --data_dir=$DATA_DIR --quant_model="quant_vww_mobilenet_prunedB13" 

exit 0

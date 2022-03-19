#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

# Attachment Location
python quantize.py --saved_model_path="ckpt_kws_aux_dscnn_hw/dscnn_block1/dscnn_block1" --tfl_file_name="quant_kws_dscnn_hw/quant_kws_dscnn_hw_auxB1_int8.tflite"  --dct_coefficient_count=10 --feature_type=lfbe
python quantize.py --saved_model_path="ckpt_kws_aux_dscnn_hw/dscnn_block2/dscnn_block2" --tfl_file_name="quant_kws_dscnn_hw/quant_kws_dscnn_hw_auxB2_int8.tflite"  --dct_coefficient_count=10 --feature_type=lfbe
python quantize.py --saved_model_path="ckpt_kws_aux_dscnn_hw/dscnn_block3/dscnn_block3" --tfl_file_name="quant_kws_dscnn_hw/quant_kws_dscnn_hw_auxB3_int8.tflite"  --dct_coefficient_count=10 --feature_type=lfbe
python quantize.py --saved_model_path="ckpt_kws_aux_dscnn_hw/dscnn_block4/dscnn_block4" --tfl_file_name="quant_kws_dscnn_hw/quant_kws_dscnn_hw_auxB4_int8.tflite"  --dct_coefficient_count=10 --feature_type=lfbe
python quantize.py --saved_model_path="ckpt_kws_aux_dscnn_hw/dscnn_conv1/dscnn_conv1" --tfl_file_name="quant_kws_dscnn_hw/quant_kws_dscnn_hw_auxC1_int8.tflite"  --dct_coefficient_count=10 --feature_type=lfbe

exit 0

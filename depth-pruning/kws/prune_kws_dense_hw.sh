#!/bin/sh

TRAIN_DIR="ckpt_kws_dense_hw_aux"
BASE_MODEL_PATH="kws_dscnn_hw"

export CUDA_VISIBLE_DEVICES=1
FROZEN_HEAD="True"
NO_PRETRAINED="False"

# Attachment Location
python prune_kws_dscnn_aux.py --attach_tensor_name=activation --aux_arch=dense --ckpt_name=dscnn_conv1 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --epochs=30 --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --dct_coefficient_count=10 --feature_type=lfbe
python prune_kws_dscnn_aux.py --attach_tensor_name=activation_2 --aux_arch=dense --ckpt_name=dscnn_block1 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --epochs=30 --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --dct_coefficient_count=10 --feature_type=lfbe
python prune_kws_dscnn_aux.py --attach_tensor_name=activation_4 --aux_arch=dense --ckpt_name=dscnn_block2 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --epochs=30 --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --dct_coefficient_count=10 --feature_type=lfbe
python prune_kws_dscnn_aux.py --attach_tensor_name=activation_6 --aux_arch=dense --ckpt_name=dscnn_block3 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --epochs=30 --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --dct_coefficient_count=10 --feature_type=lfbe
python prune_kws_dscnn_aux.py --attach_tensor_name=activation_8 --aux_arch=dense --ckpt_name=dscnn_block4 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --epochs=30 --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --dct_coefficient_count=10 --feature_type=lfbe

exit 0

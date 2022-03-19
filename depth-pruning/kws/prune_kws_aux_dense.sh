#!/bin/sh

TRAIN_DIR="ckpt_kws_aux_dense"
BASE_MODEL_PATH="kws_dscnn"

export CUDA_VISIBLE_DEVICES=1
FROZEN_HEAD="True"
NO_PRETRAINED="False"


# Attachment Location
python prune_kws_aux.py --attach_tensor_name=activation --aux_arch=dense --ckpt_name=dense_conv1 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --epochs=30 --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED
python prune_kws_aux.py --attach_tensor_name=activation_2 --aux_arch=dense --ckpt_name=dense_block1 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --epochs=30 --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED
python prune_kws_aux.py --attach_tensor_name=activation_4 --aux_arch=dense --ckpt_name=dense_block2 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --epochs=30 --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED
python prune_kws_aux.py --attach_tensor_name=activation_6 --aux_arch=dense --ckpt_name=dense_block3 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --epochs=30 --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED
python prune_kws_aux.py --attach_tensor_name=activation_8 --aux_arch=dense --ckpt_name=dense_block4 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --epochs=30 --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED

exit 0

#!/bin/sh

TRAIN_DIR="ckpt_vww_mobilenet_pruned_gray_dscnn"
BASE_MODEL_PATH="vww_96_mobilenet_gray"
DATA_DIR="/home/josen/tiny/v0.5/training/visual_wake_words/vw_coco2014_96"

FROZEN_HEAD="True"
NO_PRETRAINED="False" 
ARCH="mobilenetV1"
AUX_ARCH="dscnn32_16"

export CUDA_VISIBLE_DEVICES=1

# Sweep Attachment Location
python prune_vww_aux_gray.py --attach_tensor_name=activation --aux_arch=$AUX_ARCH --ckpt_name=dscnn_conv1 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --arch=$ARCH
python prune_vww_aux_gray.py --attach_tensor_name=activation_2 --aux_arch=$AUX_ARCH --ckpt_name=dscnn_block1 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --arch=$ARCH
python prune_vww_aux_gray.py --attach_tensor_name=activation_4 --aux_arch=$AUX_ARCH --ckpt_name=dscnn_block2 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --arch=$ARCH
python prune_vww_aux_gray.py --attach_tensor_name=activation_6 --aux_arch=$AUX_ARCH --ckpt_name=dscnn_block3 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --arch=$ARCH
python prune_vww_aux_gray.py --attach_tensor_name=activation_8 --aux_arch=$AUX_ARCH --ckpt_name=dscnn_block4 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --arch=$ARCH
python prune_vww_aux_gray.py --attach_tensor_name=activation_10 --aux_arch=$AUX_ARCH --ckpt_name=dscnn_block5 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --arch=$ARCH
python prune_vww_aux_gray.py --attach_tensor_name=activation_12 --aux_arch=$AUX_ARCH --ckpt_name=dscnn_block6 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --arch=$ARCH
python prune_vww_aux_gray.py --attach_tensor_name=activation_14 --aux_arch=$AUX_ARCH --ckpt_name=dscnn_block7 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --arch=$ARCH
python prune_vww_aux_gray.py --attach_tensor_name=activation_16 --aux_arch=$AUX_ARCH --ckpt_name=dscnn_block8 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --arch=$ARCH
python prune_vww_aux_gray.py --attach_tensor_name=activation_18 --aux_arch=$AUX_ARCH --ckpt_name=dscnn_block9 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --arch=$ARCH
python prune_vww_aux_gray.py --attach_tensor_name=activation_20 --aux_arch=$AUX_ARCH --ckpt_name=dscnn_block10 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --arch=$ARCH
python prune_vww_aux_gray.py --attach_tensor_name=activation_22 --aux_arch=$AUX_ARCH --ckpt_name=dscnn_block11 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --arch=$ARCH
python prune_vww_aux_gray.py --attach_tensor_name=activation_24 --aux_arch=$AUX_ARCH --ckpt_name=dscnn_block12 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --arch=$ARCH
python prune_vww_aux_gray.py --attach_tensor_name=activation_26 --aux_arch=$AUX_ARCH --ckpt_name=dscnn_block13 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --frozen_head=$FROZEN_HEAD --nopretrained=$NO_PRETRAINED --arch=$ARCH

exit 0

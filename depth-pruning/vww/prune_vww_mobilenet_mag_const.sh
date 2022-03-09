#!/bin/sh

TRAIN_DIR="ckpt_vww_mobilenet_mag_constant"
BASE_MODEL_PATH="vww_96_mobilenet"
DATA_DIR="/home/josen/tiny/v0.5/training/visual_wake_words/vw_coco2014_96"

FROZEN_HEAD="True"
NO_PRETRAINED="False" 
ARCH="mobilenetV1"
AUX_ARCH="dscnn32_16"

PRUNING_SCHEME="constant"

export CUDA_VISIBLE_DEVICES=1

# Attachment Location
python prune_vww_mag.py --ckpt_name=mag_constant_10 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --end_sparsity=0.10 --epochs=20 --pruning_scheme=$PRUNING_SCHEME
python prune_vww_mag.py --ckpt_name=mag_constant_15 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --end_sparsity=0.15 --epochs=20 --pruning_scheme=$PRUNING_SCHEME
python prune_vww_mag.py --ckpt_name=mag_constant_20 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --end_sparsity=0.20 --epochs=20 --pruning_scheme=$PRUNING_SCHEME
python prune_vww_mag.py --ckpt_name=mag_constant_25 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --end_sparsity=0.25 --epochs=20 --pruning_scheme=$PRUNING_SCHEME
python prune_vww_mag.py --ckpt_name=mag_constant_30 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --end_sparsity=0.30 --epochs=20 --pruning_scheme=$PRUNING_SCHEME
python prune_vww_mag.py --ckpt_name=mag_constant_35 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --end_sparsity=0.35 --epochs=20 --pruning_scheme=$PRUNING_SCHEME
python prune_vww_mag.py --ckpt_name=mag_constant_40 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --end_sparsity=0.40 --epochs=20 --pruning_scheme=$PRUNING_SCHEME
python prune_vww_mag.py --ckpt_name=mag_constant_45 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --end_sparsity=0.45 --epochs=20 --pruning_scheme=$PRUNING_SCHEME
python prune_vww_mag.py --ckpt_name=mag_constant_50 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --end_sparsity=0.50 --epochs=20 --pruning_scheme=$PRUNING_SCHEME
python prune_vww_mag.py --ckpt_name=mag_constant_55 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --end_sparsity=0.55 --epochs=20 --pruning_scheme=$PRUNING_SCHEME
python prune_vww_mag.py --ckpt_name=mag_constant_60 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --end_sparsity=0.60 --epochs=20 --pruning_scheme=$PRUNING_SCHEME
python prune_vww_mag.py --ckpt_name=mag_constant_65 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --end_sparsity=0.65 --epochs=20 --pruning_scheme=$PRUNING_SCHEME
python prune_vww_mag.py --ckpt_name=mag_constant_70 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --end_sparsity=0.70 --epochs=20 --pruning_scheme=$PRUNING_SCHEME
python prune_vww_mag.py --ckpt_name=mag_constant_75 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --end_sparsity=0.75 --epochs=20 --pruning_scheme=$PRUNING_SCHEME
python prune_vww_mag.py --ckpt_name=mag_constant_80 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --end_sparsity=0.80 --epochs=20 --pruning_scheme=$PRUNING_SCHEME
python prune_vww_mag.py --ckpt_name=mag_constant_85 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --end_sparsity=0.85 --epochs=20 --pruning_scheme=$PRUNING_SCHEME
python prune_vww_mag.py --ckpt_name=mag_constant_90 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --end_sparsity=0.90 --epochs=20 --pruning_scheme=$PRUNING_SCHEME
python prune_vww_mag.py --ckpt_name=mag_constant_95 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --data_dir=$DATA_DIR --end_sparsity=0.95 --epochs=20 --pruning_scheme=$PRUNING_SCHEME

exit 0

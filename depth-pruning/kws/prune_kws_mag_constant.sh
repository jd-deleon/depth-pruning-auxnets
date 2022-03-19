#!/bin/sh

TRAIN_DIR="ckpt_kws_mag_constant"
BASE_MODEL_PATH="kws_dscnn"
# DATA_DIR="/home/josen/tiny/v0.5/training/visual_wake_words/vw_coco2014_96"

PRUNING_SCHEME="constant"

export CUDA_VISIBLE_DEVICES=0

# Attachment Location
python prune_kws_mag.py --ckpt_name=mag_10 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --end_sparsity=0.10 --epochs=30 --pruning_scheme=$PRUNING_SCHEME
python prune_kws_mag.py --ckpt_name=mag_15 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --end_sparsity=0.15 --epochs=30 --pruning_scheme=$PRUNING_SCHEME
python prune_kws_mag.py --ckpt_name=mag_20 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --end_sparsity=0.20 --epochs=30 --pruning_scheme=$PRUNING_SCHEME
python prune_kws_mag.py --ckpt_name=mag_25 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --end_sparsity=0.25 --epochs=30 --pruning_scheme=$PRUNING_SCHEME
python prune_kws_mag.py --ckpt_name=mag_30 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --end_sparsity=0.30 --epochs=30 --pruning_scheme=$PRUNING_SCHEME
python prune_kws_mag.py --ckpt_name=mag_35 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --end_sparsity=0.35 --epochs=30 --pruning_scheme=$PRUNING_SCHEME
python prune_kws_mag.py --ckpt_name=mag_40 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --end_sparsity=0.40 --epochs=30 --pruning_scheme=$PRUNING_SCHEME
python prune_kws_mag.py --ckpt_name=mag_45 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --end_sparsity=0.45 --epochs=30 --pruning_scheme=$PRUNING_SCHEME
python prune_kws_mag.py --ckpt_name=mag_50 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --end_sparsity=0.50 --epochs=30 --pruning_scheme=$PRUNING_SCHEME
python prune_kws_mag.py --ckpt_name=mag_55 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --end_sparsity=0.55 --epochs=30 --pruning_scheme=$PRUNING_SCHEME
python prune_kws_mag.py --ckpt_name=mag_60 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --end_sparsity=0.60 --epochs=30 --pruning_scheme=$PRUNING_SCHEME
python prune_kws_mag.py --ckpt_name=mag_65 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --end_sparsity=0.65 --epochs=30 --pruning_scheme=$PRUNING_SCHEME
python prune_kws_mag.py --ckpt_name=mag_70 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --end_sparsity=0.70 --epochs=30 --pruning_scheme=$PRUNING_SCHEME
python prune_kws_mag.py --ckpt_name=mag_75 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --end_sparsity=0.75 --epochs=30 --pruning_scheme=$PRUNING_SCHEME
python prune_kws_mag.py --ckpt_name=mag_80 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --end_sparsity=0.80 --epochs=30 --pruning_scheme=$PRUNING_SCHEME
python prune_kws_mag.py --ckpt_name=mag_85 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --end_sparsity=0.85 --epochs=30 --pruning_scheme=$PRUNING_SCHEME
python prune_kws_mag.py --ckpt_name=mag_90 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --end_sparsity=0.90 --epochs=30 --pruning_scheme=$PRUNING_SCHEME
python prune_kws_mag.py --ckpt_name=mag_95 --train_dir=$TRAIN_DIR --base_model_path=$BASE_MODEL_PATH --end_sparsity=0.95 --epochs=30 --pruning_scheme=$PRUNING_SCHEME

exit 0

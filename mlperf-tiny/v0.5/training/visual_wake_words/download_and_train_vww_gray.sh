
TRAIN_DIR="trained_models/vww_96_mobilenetV1-gray"
ARCH="mobilenetV1"

export CUDA_VISIBLE_DEVICES=0

#!/bin/bash
# Downoad the dataset.
wget https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz
tar -xvf vw_coco2014_96.tar.gz

# Preprocess the dataset and train/convert the VWW model.
#python3 parse_coco.py annotations/instances_train2017.json
python3 train_vww_gray.py --arch=$ARCH --train_dir=$TRAIN_DIR
python3 convert_vww.py $TRAIN_DIR

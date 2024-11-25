#!/bin/bash

# 실행할 Python 스크립트
SCRIPT_NAME="VIT_BASELINE.py"

# 실행에 필요한 옵션 및 인자
EXP_NAME="Task01_BrainTumour_2"
GPU_IDS="0"
DATA_DIR="./dataset/Task01_BrainTumour"
DATA_FILE="./dataset.json"
LR="1e-5"
EPOCHS="100"
BATCH_SIZE="64"
IMAGE_SIZE="128,128,80"
PATCH_SIZE="16,16,16"

# Python 명령 실행
python $SCRIPT_NAME \
    --exp_name $EXP_NAME \
    --gpu_ids $GPU_IDS \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --lr $LR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --image_size $IMAGE_SIZE \
    --patch_size $PATCH_SIZE

#!/bin/bash

dataset=$1
device=${2:-cuda}
dtype=${3:-float32}

python misc_scripts/save_image_feature.py \
    --cache_dir data/cache \
    --dataset $dataset \
    --model_name sam_vitb \
    --device $device \
    --batch_size 1 \
    --dtype $dtype \
    --ckpt ckpt/sam_vit_b_01ec64.pth \
    --shard_size 512
    

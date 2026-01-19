#!/bin/bash

dataset=$1

python misc_scripts/save_image_feature.py \
    --cache_dir data/cache \
    --dataset $dataset \
    --model_name sam_vitb \
    --device cuda \
    --batch_size 1 \
    --dtype float32 \
    --ckpt ckpt/sam_vit_b_01ec64.pth
    
#!/bin/bash

python misc_scripts/save_image_feature.py \
    --cache_dir data/cache \
    --dataset hqseg44k \
    --model_name mobile_sam \
    --device cuda \
    --batch_size 4 \
    --dtype float32 \
    --ckpt ckpt/mobile_sam.pt
    
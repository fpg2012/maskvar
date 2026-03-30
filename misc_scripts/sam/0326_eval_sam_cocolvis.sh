#!/bin/bash
# Evaluate SAM on COCO-LVIS validation set

DEVICE=$1

python misc_scripts/eval_sam.py \
    --device $DEVICE \
    --sam_checkpoint ckpt/sam_vit_b_01ec64.pth \
    --vqvae vqvae_single_5_stages_v1 \
    --vqvae_checkpoint out/out_vqvae_5_stages_v1/ckpt/vqvae_single_epoch_50.pth \
    --dataset cocolvis \
    --dataset_split val \
    --batch_size 8 \
    --num_clicks 1 \
    --outdir out/sam_eval_cocolvis \
    --image_feature_cache_dir data/cache \
    --visualize

#!/bin/bash

DEVICE=${1:-cuda}
OUTDIR=${2:-out/train_simple_var_sampe_v0_lr1e-3_bs32}

python train_scripts/train_simple_var.py \
    --device $DEVICE \
    --outdir $OUTDIR \
    --lr 1e-3 \
    --batch_size 16 \
    --accumulate_steps 2 \
    --outer_iters 20 \
    --val_iters 64 \
    --inner_iters 5000 \
    --use_sam_pe \
    --prompt_encoder_checkpoint ckpt/sam_vit_b_01ec64.pth \
    --image_feature_cache_dir data/cache \
    --image_encoder sam_vitb \
    --dtype bfloat16
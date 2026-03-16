#!/bin/bash

# random init, not use sam checkpoint

DEVICE=${1:-cuda}
OUTDIR=${2:-out/debug_simple_var_sd_random_init_sampe}

python train_scripts/train_simple_var.py \
    --device $DEVICE \
    --outdir $OUTDIR \
    --lr 1e-3 \
    --batch_size 16 \
    --accumulate_steps 2 \
    --simple_var simple_var_sd \
    --use_dummy_dataset_for_debug \
    --use_sam_pe \
    --outer_iters 2 \
    --val_iters -1 \
    --inner_iters 500 \
    --image_feature_cache_dir data/cache \
    --image_encoder sam_vitb \
    --dtype bfloat16 \
    --use_sam_pe \
    --prompt_encoder_checkpoint ckpt/sam_vit_b_01ec64.pth \
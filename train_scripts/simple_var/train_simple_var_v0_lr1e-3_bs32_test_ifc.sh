#!/bin/bash

DEVICE=${1:-cuda}
OUTDIR=${2:-out/simple_var_v0_lr1e-3_bs32_test_ifc}

python train_scripts/train_simple_var.py \
    --device $DEVICE \
    --outdir $OUTDIR \
    --lr 1e-3 \
    --batch_size 16 \
    --accumulate_steps 2 \
    --outer_iters 5 \
    --val_iters 64 \
    --inner_iters 64 \
    --image_feature_cache_dir data/cache \
    --image_encoder sam_vitb
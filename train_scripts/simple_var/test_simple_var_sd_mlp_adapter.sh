#!/bin/bash

DEVICE=${1:-cuda}
OUTDIR=${2:-out/debug_simple_var_sd_mlp_adapter}

python train_scripts/train_simple_var.py \
    --device $DEVICE \
    --outdir $OUTDIR \
    --lr 1e-3 \
    --batch_size 16 \
    --accumulate_steps 2 \
    --simple_var simple_var_sd_mlp_adapter \
    --simple_var_init_checkpoint ckpt/simple_var_sd_mlp_adapter_init.pth \
    --use_sam_pe \
    --use_dummy_dataset_for_debug \
    --outer_iters 2 \
    --val_iters -1 \
    --inner_iters 1000 \
    --image_feature_cache_dir data/cache \
    --image_encoder sam_vitb \
    --dtype bfloat16
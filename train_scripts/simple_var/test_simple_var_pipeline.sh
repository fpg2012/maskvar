#!/bin/bash
# Test script for SimpleVAR training pipeline
# Uses dummy dataset for quick testing

# Default values
DEVICE=${1:-cuda:0}
OUTDIR=${2:-out/test_simple_var_pipeline}

# Use single GPU, small batch, few iterations for quick testing
python train_scripts/train_simple_var.py \
    --device $DEVICE \
    --outdir $OUTDIR \
    --lr 2e-3 \
    --batch_size 4 \
    --accumulate_steps 1 \
    --outer_iters 2 \
    --val_iters 2 \
    --inner_iters 8 \
    --use_sam_pe \
    --enable_clicks \
    --prompt_encoder_checkpoint ckpt/sam_vit_b_01ec64.pth \
    --image_feature_cache_dir data/cache \
    --image_encoder sam_vitb \
    --dtype bfloat16 \
    --dataset coconut_hf \
    --dl_workers 2 \
    --prefetch_factor 1 \
    --use_dummy_dataset_for_debug

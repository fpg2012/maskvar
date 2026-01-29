#!/bin/bash
N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_train_test_bs32}
export MASTER_PORT=${3:-29500}

torchrun --nproc_per_node=$N_NODE train_scripts/train_simple_var.py \
    --outdir $OUTDIR \
    --lr 2e-3 \
    --batch_size 32 \
    --accumulate_steps 1 \
    --outer_iters 1 \
    --val_iters 0 \
    --inner_iters 0 \
    --use_sam_pe \
    --prompt_encoder_checkpoint ckpt/sam_vit_b_01ec64.pth \
    --image_feature_cache_dir data/cache \
    --image_encoder sam_vitb \
    --dtype bfloat16 \
    --dataset hqseg44k \
    --dl_workers 16 \
    --prefetch_factor 2
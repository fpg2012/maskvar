#!/bin/bash
# Test/Overfitting script for SimpleMaskVqvae on HQSeg44k
# Used to verify the training pipeline works correctly

N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_simple_mask_vqvae_test}
export MASTER_PORT=${3:-29500}
export OMP_NUM_THREADS=4

torchrun --nproc_per_node=$N_NODE train_scripts/train_simple_mask_vqvae.py \
    --out_dir $OUTDIR \
    --outer_iters 8 \
    --inner_iters 2048 \
    --val_iters 128 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --accumulate_steps 1 \
    --num_workers 4 \
    --prefetch_factor 1 \
    --dataset hqseg44k \
    --sam_checkpoint_path ckpt/mobile_sam.pt \
    --dtype bfloat16 \
    --debug \
    --debug_iters 100 \
    --no_compile

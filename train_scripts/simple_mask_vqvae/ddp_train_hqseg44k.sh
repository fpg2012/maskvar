#!/bin/bash
# Train SimpleMaskVqvae on HQSeg44k dataset
# This is the main training script for high-quality segmentation

N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_simple_mask_vqvae_hqseg44k_ep10_12dice_nfl}
export MASTER_PORT=${3:-29500}
export OMP_NUM_THREADS=4

if [ $N_NODE -eq 0 ]; then
    python train_scripts/train_simple_mask_vqvae.py \
    --out_dir $OUTDIR \
    --outer_iters 10 \
    --inner_iters 0 \
    --val_iters 0 \
    --batch_size 8 \
    --learning_rate 1e-3 \
    --accumulate_steps 2 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --dataset hqseg44k \
    --sam_checkpoint_path ckpt/mobile_sam.pt \
    --dtype bfloat16 \
    --no_compile \
    --freeze_image_encoder \
    --freeze_mask_encoder \
    --loss dicenfl
else
    torchrun --nproc_per_node=$N_NODE train_scripts/train_simple_mask_vqvae.py \
    --out_dir $OUTDIR \
    --outer_iters 10 \
    --inner_iters 0 \
    --val_iters 0 \
    --batch_size 8 \
    --learning_rate 1e-3 \
    --accumulate_steps 2 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --dataset hqseg44k \
    --sam_checkpoint_path ckpt/mobile_sam.pt \
    --dtype bfloat16 \
    --no_compile \
    --freeze_image_encoder \
    --freeze_mask_encoder
fi

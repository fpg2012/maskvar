#!/bin/bash
# Train SimpleMaskVqvae on HQSeg44k dataset
# This is the main training script for high-quality segmentation

N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_simple_mask_vqvae_hqseg44k_ep10_13dinov3}
export MASTER_PORT=${3:-29500}
export OMP_NUM_THREADS=4

if [ $N_NODE -eq 0 ]; then
    python train_scripts/train_simple_mask_vqvae.py \
    --out_dir $OUTDIR \
    --outer_iters 10 \
    --inner_iters 0 \
    --val_iters 0 \
    --batch_size 16 \
    --learning_rate 2e-3 \
    --accumulate_steps 1 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --dataset hqseg44k \
    --config simple_mask_vqvae_dim384 \
    --image_encoder_checkpoint ckpt/dino_v3_vits.safetensors \
    --image_encoder_config dino_v3_vits \
    --dtype bfloat16 \
    --no_compile \
    --freeze_image_encoder \
    --loss dicenfl
else
    torchrun --nproc_per_node=$N_NODE train_scripts/train_simple_mask_vqvae.py \
    --out_dir $OUTDIR \
    --outer_iters 10 \
    --inner_iters 0 \
    --val_iters 0 \
    --batch_size 16 \
    --learning_rate 2e-3 \
    --accumulate_steps 1 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --dataset hqseg44k \
    --config simple_mask_vqvae_dim384 \
    --image_encoder_checkpoint ckpt/dino_v3_vits.safetensors \
    --image_encoder_config dino_v3_vits \
    --dtype bfloat16 \
    --no_compile \
    --freeze_image_encoder \
    --loss dicenfl
fi

#!/bin/bash
# Train MaskVQVAE on HQSeg44k dataset

N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_mask_vqvae_hqseg44k_ep50_v0}
export MASTER_PORT=${3:-29500}
export OMP_NUM_THREADS=4

torchrun --nproc_per_node=$N_NODE train_scripts/train_mask_vqvae.py \
    --out_dir $OUTDIR \
    --dataset hqseg44k \
    --image_feature_cache_dir data/features/sam_vitb \
    --image_encoder sam_vitb \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_workers 4 \
    --use_focal_loss

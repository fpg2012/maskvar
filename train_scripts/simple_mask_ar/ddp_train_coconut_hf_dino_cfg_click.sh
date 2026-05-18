#!/bin/bash
# Train CFG + click-conditioned SimpleMaskAR on COCONut HF dataset.
# Initializes from an already trained non-click SimpleMaskAR checkpoint.
#
# Usage:
#   bash train_scripts/simple_mask_ar/ddp_train_coconut_hf_dino_cfg_click.sh [N_NODE] [OUTDIR] [VQVAE_CKPT] [AR_CKPT] [MASTER_PORT]
#
# Example:
#   bash train_scripts/simple_mask_ar/ddp_train_coconut_hf_dino_cfg_click.sh 4 \
#     out/ddp_simple_mask_ar_coconut_cfg_click_ep5 \
#     out/ddp_simple_mask_vqvae_coconut_ep5_19bugfix_retrain/checkpoints/latest.pth \
#     out/ddp_simple_mask_ar_coconut_ep10/checkpoints/latest.pth

N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_simple_mask_ar_coconut_cfg_click_ep5}
VQVAE_CKPT=${3:-out/ddp_simple_mask_vqvae_coconut_ep5_19bugfix_retrain/checkpoints/latest.pth}
AR_CKPT=${4:-out/ddp_simple_mask_ar_coconut_ep10/checkpoints/latest.pth}
export MASTER_PORT=${5:-29500}
export OMP_NUM_THREADS=4

COMMON_ARGS=(
    --out_dir "$OUTDIR"
    --vqvae_checkpoint "$VQVAE_CKPT"
    --vqvae_config simple_mask_vqvae_dim384
    --vqvae_image_encoder_checkpoint ckpt/dino_v3_vits.safetensors
    --vqvae_image_encoder_config dino_v3_vits
    --checkpoint "$AR_CKPT"
    --enable_click
    --enable_cfg
    --cfg_drop_click_prob 0.1
    --outer_iters 5
    --inner_iters 0
    --val_iters 0
    --batch_size 16
    --learning_rate 2e-4
    --accumulate_steps 1
    --num_workers 8
    --prefetch_factor 4
    --dataset coconut_hf
    --config simple_mask_ar
    --dtype bfloat16
    --train_subset_index data/subset/coconut_hf_train-25_percent.npy
    --log_interval 128
)

if [ "$N_NODE" -eq 0 ]; then
    python train_scripts/train_simple_mask_ar.py "${COMMON_ARGS[@]}" --no_compile
else
    torchrun --nproc_per_node="$N_NODE" train_scripts/train_simple_mask_ar.py \
        "${COMMON_ARGS[@]}" \
        --disable_find_unused_parameters
fi

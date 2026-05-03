#!/bin/bash
# Train SimpleMaskVAR on top of a frozen multiscale V2 SimpleMaskVqvae.
# Usage:
#   bash train_scripts/simple_mask_ar/ddp_train_var_multiscale_v2_coconut_hf_dino.sh [N_NODE] [OUTDIR] [VQVAE_CKPT] [MASTER_PORT]

N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_simple_mask_var_multiscale_v2_coconut_ep10}
VQVAE_CKPT=${3:-out/ddp_simple_mask_vqvae_multiscale_v2_coconut_ep10/checkpoints/latest.pth}
export MASTER_PORT=${4:-29500}
export OMP_NUM_THREADS=4

COMMON_ARGS=(
    --out_dir "$OUTDIR"
    --vqvae_checkpoint "$VQVAE_CKPT"
    --vqvae_config simple_mask_vqvae_multiscale_v2_dim384
    --vqvae_image_encoder_checkpoint ckpt/dino_v3_vits.safetensors
    --vqvae_image_encoder_config dino_v3_vits
    --outer_iters 10
    --inner_iters 0
    --val_iters 0
    --batch_size 16
    --learning_rate 2e-4
    --accumulate_steps 1
    --num_workers 8
    --prefetch_factor 4
    --dataset coconut_hf
    --config simple_mask_var
    --dtype bfloat16
    --train_subset_index data/subset/coconut_hf_train-25_percent.npy
    --log_interval 128
)

if [ "$N_NODE" -eq 0 ]; then
    python train_scripts/train_simple_mask_var.py "${COMMON_ARGS[@]}" --no_compile
else
    torchrun --nproc_per_node="$N_NODE" --master_port="$MASTER_PORT" \
        train_scripts/train_simple_mask_var.py "${COMMON_ARGS[@]}" \
        --disable_find_unused_parameters
fi

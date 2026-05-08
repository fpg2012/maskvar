#!/bin/bash
# Smoke test for SimpleMaskVARV2 using the dummy dataset path.
# This avoids the sharded sampler and disables torch.compile.
# Usage:
#   bash train_scripts/simple_mask_ar/ddp_train_var_v2_smoketest_coconut_hf_dino.sh [N_NODE] [OUTDIR] [VQVAE_CKPT] [MASTER_PORT]

N_NODE=${1:-0}
OUTDIR=${2:-out/ddp_simple_mask_var_v2_smoketest_coconut_ep10}
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
    --inner_iters 2000
    --val_iters 4
    --batch_size 2
    --learning_rate 2e-4
    --accumulate_steps 1
    --num_workers 0
    --prefetch_factor 2
    --dataset coconut_hf
    --config simple_mask_var_v2
    --dtype bfloat16
    --debug
    --debug_iters 5000
    --log_interval 1
)

if [ "$N_NODE" -eq 0 ]; then
    python train_scripts/train_simple_mask_var.py "${COMMON_ARGS[@]}"
else
    torchrun --nproc_per_node="$N_NODE" --master_port="$MASTER_PORT" \
        train_scripts/train_simple_mask_var.py "${COMMON_ARGS[@]}" \
        --disable_find_unused_parameters
fi

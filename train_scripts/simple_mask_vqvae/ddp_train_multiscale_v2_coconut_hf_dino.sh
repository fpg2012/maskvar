#!/bin/bash
# Train VAR-style residual multi-scale SimpleMaskVqvae on COCONut HF dataset.

N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_simple_mask_vqvae_multiscale_v2_coconut_ep10}
export MASTER_PORT=${3:-29500}
INIT_CKPT=${4:-out/ddp_simple_mask_vqvae_coconut_ep5_19bugfix_retrain/checkpoints/latest.pth}
export OMP_NUM_THREADS=4

COMMON_ARGS=(
    --out_dir "$OUTDIR"
    --outer_iters 10
    --inner_iters 0
    --val_iters 0
    --batch_size 16
    --learning_rate 2e-4
    --accumulate_steps 1
    --num_workers 8
    --prefetch_factor 4
    --dataset coconut_hf
    --config simple_mask_vqvae_multiscale_v2_dim384
    --image_encoder_checkpoint ckpt/dino_v3_vits.safetensors
    --image_encoder_config dino_v3_vits
    --dtype bfloat16
    --loss dicenfl
    --enable_vq
    --freeze_image_encoder
    --log_interval 64
    --train_subset_index data/subset/coconut_hf_train-25_percent.npy
    --vq_loss_weight 0.25
    --disable_find_unused_parameters
)

if [ -n "$INIT_CKPT" ]; then
    COMMON_ARGS+=(--checkpoint "$INIT_CKPT")
fi

if [ "$N_NODE" -eq 0 ]; then
    python train_scripts/train_simple_mask_vqvae.py "${COMMON_ARGS[@]}"
else
    torchrun --nproc_per_node="$N_NODE" --master_port="$MASTER_PORT" train_scripts/train_simple_mask_vqvae.py \
        "${COMMON_ARGS[@]}"
fi

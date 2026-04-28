#!/bin/bash
# Train shape-only SimpleMaskVqvae with VQ on COCONut HF.
# Args:
#   $1: number of gpus on this node; use 0 for single-process python
#   $2: output directory
#   $3: master port
#   $4: optional init checkpoint
#   $5: optional kmeans centroids checkpoint

N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_simple_mask_vqvae_shape_only_vq_coconut}
export MASTER_PORT=${3:-29500}
INIT_CKPT=${4:-}
KMEANS_CENTROIDS=${5:-}
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
    --config simple_mask_vqvae_shape_only_dim384
    --dtype bfloat16
    --loss dicenfl
    --train_subset_index data/subset/coconut_hf_train-25_percent.npy
    --vq_loss_weight 0.25
    --log_interval 128
)

if [ -n "$INIT_CKPT" ]; then
    COMMON_ARGS+=(--checkpoint "$INIT_CKPT")
fi

if [ -n "$KMEANS_CENTROIDS" ]; then
    COMMON_ARGS+=(--kmeans_centroids "$KMEANS_CENTROIDS")
fi

if [ "$N_NODE" -eq 0 ]; then
    python train_scripts/train_simple_mask_vqvae.py "${COMMON_ARGS[@]}" --no_compile
else
    torchrun --nproc_per_node="$N_NODE" train_scripts/train_simple_mask_vqvae.py \
        "${COMMON_ARGS[@]}" \
        --no_compile
fi

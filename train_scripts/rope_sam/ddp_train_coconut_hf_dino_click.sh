#!/bin/bash
# Train RopeSAM on COCONut HF dataset with DINO v3 image features and click prompts.
#
# Usage:
#   bash train_scripts/rope_sam/ddp_train_coconut_hf_dino_click.sh [N_NODE] [OUTDIR] [MASTER_PORT] [INIT_CKPT]

N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_rope_sam_coconut_hf_dino_click}
export MASTER_PORT=${3:-29500}
INIT_CKPT=${4:-}
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
    --config rope_sam_dim384
    --image_encoder_config dino_v3_vits
    --image_encoder_checkpoint ckpt/dino_v3_vits.safetensors
    --freeze_image_encoder
    --max_clicks 10
    --interactive_click_warmup_iters 10000
    --loss nfl
    --dtype bfloat16
    --train_subset_index data/subset/coconut_hf_train-25_percent.npy
    --log_interval 128
)

if [ -n "$INIT_CKPT" ]; then
    COMMON_ARGS+=(--checkpoint "$INIT_CKPT")
fi

if [ "$N_NODE" -eq 0 ]; then
    python train_scripts/train_rope_sam.py "${COMMON_ARGS[@]}" --no_compile
else
    torchrun --nproc_per_node="$N_NODE" train_scripts/train_rope_sam.py \
        "${COMMON_ARGS[@]}" \
        --disable_find_unused_parameters
fi

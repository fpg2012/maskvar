#!/bin/bash
# Fine-tune SAM ViT-B mask decoder on COCONut HF. The image encoder and prompt
# encoder are frozen; checkpoints/latest.pth is a full SAM state_dict.
#
# Usage:
#   bash train_scripts/sam/ddp_train_decoder_coconut_hf_click.sh [N_NODE] [OUTDIR] [MASTER_PORT] [INIT_OR_RESUME]

set -euo pipefail

N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_sam_decoder_coconut_hf_click}
export MASTER_PORT=${3:-29502}
RESUME_FROM=${4:-}
export OMP_NUM_THREADS=4

COMMON_ARGS=(
    --out_dir "$OUTDIR"
    --sam_checkpoint ckpt/sam_vit_b_01ec64.pth
    --sam_model_type vit_b
    --dataset coconut_hf
    --outer_iters 5
    --inner_iters 0
    --val_iters -1
    --batch_size 1
    --learning_rate 1e-4
    --weight_decay 0.01
    --max_clicks 10
    --train_max_clicks 1
    --use_prev_mask
    --loss nfl
    --dtype bfloat16
    --num_workers 8
    --prefetch_factor 4
    --image_feature_cache_dir data/cache
    --image_feature_cache_max_shard 32
    --train_subset_index data/subset/coconut_hf_train-25_percent.npy
    --log_interval 50
)

if [ -n "$RESUME_FROM" ]; then
    COMMON_ARGS+=(--resume_from "$RESUME_FROM")
fi

if [ "$N_NODE" -eq 0 ]; then
    python train_scripts/train_sam_decoder.py "${COMMON_ARGS[@]}"
else
    torchrun --nproc_per_node="$N_NODE" train_scripts/train_sam_decoder.py \
        "${COMMON_ARGS[@]}"
fi

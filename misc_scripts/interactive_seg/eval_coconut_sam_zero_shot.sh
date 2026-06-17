#!/bin/bash
# Re-evaluate zero-shot SAM ViT-B on COCONut HF interactive segmentation.
# Defaults intentionally match the previous eval_interactive_seg.py protocol:
# no feature cache, float32 autocast disabled, no first-click multimask.
#
# Usage:
#   bash misc_scripts/interactive_seg/eval_coconut_sam_zero_shot.sh [OUTDIR] [VAL_ITERS] [DEVICE]

set -euo pipefail

OUTDIR=${1:-out/interactive_eval_coconut_sam_zero_shot}
VAL_ITERS=${2:-0}
DEVICE=${3:-cuda}

COMMON_ARGS=(
    --model sam
    --device "$DEVICE"
    --outdir "$OUTDIR"
    --dataset coconut_hf
    --dataset_split val
    --sam_checkpoint ckpt/sam_vit_b_01ec64.pth
    --sam_model_type vit_b
    --max_clicks 10
    --batch_size 1
    --num_workers 4
    --val_iters "$VAL_ITERS"
    --dtype float32
)

if [ "${USE_SAM_CACHE:-0}" -eq 1 ]; then
    COMMON_ARGS+=(
        --image_feature_cache_dir data/cache
        --sam_cache_model_name sam_vitb
        --image_feature_cache_max_shard 32
    )
fi

if [ "${SAM_MULTIMASK_FIRST_CLICK:-0}" -eq 1 ]; then
    COMMON_ARGS+=(--sam_multimask_first_click)
fi

python misc_scripts/eval_interactive_seg.py "${COMMON_ARGS[@]}"

#!/bin/bash
# Compare zero-shot SAM and coconut-trained RopeSAM on HQSeg44K.
#
# Usage:
#   bash misc_scripts/interactive_seg/eval_hqseg44k_sam_vs_rope_sam.sh [OUTDIR] [ROPE_SAM_CKPT] [VAL_ITERS] [DEVICE]

set -euo pipefail

OUTDIR=${1:-out/interactive_eval_hqseg44k_sam_vs_rope_sam}
ROPE_SAM_CKPT=${2:-out/ddp_rope_sam_coconut_hf_dino_click/checkpoints/latest.pth}
VAL_ITERS=${3:-0}
DEVICE=${4:-cuda}

python misc_scripts/eval_interactive_seg.py \
    --model both \
    --device "$DEVICE" \
    --outdir "$OUTDIR" \
    --dataset hqseg44k \
    --dataset_split val \
    --sam_checkpoint ckpt/sam_vit_b_01ec64.pth \
    --sam_model_type vit_b \
    --sam_multimask_first_click \
    --rope_sam_checkpoint "$ROPE_SAM_CKPT" \
    --rope_sam_config rope_sam_dim384 \
    --image_encoder_checkpoint ckpt/dino_v3_vits.safetensors \
    --image_encoder_config dino_v3_vits \
    --max_clicks 10 \
    --batch_size 1 \
    --num_workers 4 \
    --val_iters "$VAL_ITERS" \
    --dtype bfloat16

#!/bin/bash
# Compare coconut-finetuned SAM decoder and coconut-trained RopeSAM.
#
# Usage:
#   bash misc_scripts/interactive_seg/eval_coconut_finetuned_sam_vs_rope_sam.sh [OUTDIR] [SAM_FT_CKPT] [ROPE_SAM_CKPT] [VAL_ITERS] [DEVICE]

set -euo pipefail

OUTDIR=${1:-out/interactive_eval_coconut_sam_ft_vs_rope_sam}
SAM_FT_CKPT=${2:-out/ddp_sam_decoder_coconut_hf_click/checkpoints/latest.pth}
ROPE_SAM_CKPT=${3:-out/ddp_rope_sam_coconut_hf_dino_click/checkpoints/latest.pth}
VAL_ITERS=${4:-0}
DEVICE=${5:-cuda}

python misc_scripts/eval_interactive_seg.py \
    --model both \
    --device "$DEVICE" \
    --outdir "$OUTDIR" \
    --dataset coconut_hf \
    --dataset_split val \
    --sam_checkpoint "$SAM_FT_CKPT" \
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

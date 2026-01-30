#!/bin/bash

DEVICE=$1
OUTDIR=$2
ITERS=${3:-400000}
DATASET=$4
SAM_PE=$5

# if SAM_PE=--use_sam_pe
if [ "$SAM_PE" = "--use_sam_pe" ]; then
    USE_SAM_PE="--use_sam_pe"
else
    USE_SAM_PE=""
fi


python misc_scripts/eval_simple_var.py \
    -c simple_var \
    --device $DEVICE \
    --outdir $OUTDIR \
    --batch_size 4 \
    --checkpoint $OUTDIR/checkpoints/.simple_var.${ITERS}.pt \
    --image_feature_cache_dir data/cache \
    --image_encoder sam_vitb \
    --image_encoder_checkpoint ckpt/sam_vit_b_01ec64.pth \
    --dataset ${DATASET} \
    --dataset_split val \
    --visualize \
    $USE_SAM_PE
#!/bin/bash

DEVICE=$1
OUTDIR=$2
ITERS=${3:-400000}
DATASET=$4

python misc_scripts/eval_simple_var.py \
    -c simple_var \
    --device $DEVICE \
    --outdir $OUTDIR \
    --batch_size 4 \
    --checkpoint $OUTDIR/checkpoints/.simple_var.${ITERS}.pt \
    --image_feature_cache_dir data/cache \
    --image_encoder sam_vitb \
    --dataset ${DATASET} \
    --dataset_split val \
    --visualize
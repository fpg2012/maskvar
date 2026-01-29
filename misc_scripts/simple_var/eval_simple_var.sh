#!/bin/bash

DEVICE=$1
OUTDIR=$2
ITERS=${3:-400000}

python misc_scripts/eval_simple_var.py \
    -c simple_var \
    --device $DEVICE \
    --outdir out/$OUTDIR \
    --batch_size 8 \
    --checkpoint out/$OUTDIR/checkpoints/.simple_var.${ITERS}.pt \
    --visualize
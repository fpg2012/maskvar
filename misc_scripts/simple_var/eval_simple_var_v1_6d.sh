#!/bin/bash

DEVICE=$1
OUTDIR=$2

python misc_scripts/eval_simple_var.py \
    -c simple_var_6d \
    --device $DEVICE \
    --outdir out/$OUTDIR \
    --batch_size 8 \
    --checkpoint out/$OUTDIR/checkpoints/.simple_var.400000.pt \
    --visualize
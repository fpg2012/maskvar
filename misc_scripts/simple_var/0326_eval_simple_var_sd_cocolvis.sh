#!/bin/bash
DEVICE=$1
OUTDIR="ddp_simple_var_sd_cocolvis_v0_lr2e-3_bs32_sampe"

python misc_scripts/eval_simple_var.py \
    --device $DEVICE \
    --simple_var simple_var_sd \
    --outdir out/${OUTDIR} \
    --checkpoint out/${OUTDIR}/checkpoints/.simple_var.475840.pt \
    --batch_size 8 \
    --use_sam_pe \
    --enable_clicks \
    --dataset_split val \
    --image_feature_cache_dir data/cache \
    --dataset cocolvis \
    --image_encoder sam_vitb \
    --visualize 
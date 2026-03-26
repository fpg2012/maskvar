#!/bin/bash
N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_simple_var_sd_cocolvis_v0_lr2e-3_bs32_sampe_clicks}
export MASTER_PORT=${3:-29500}
export OMP_NUM_THREADS=4

torchrun --nproc_per_node=$N_NODE train_scripts/train_simple_var.py \
    --outdir $OUTDIR \
    --simple_var simple_var_sd \
    --simple_var_init_checkpoint ckpt/simple_var_sd_init_2.pth \
    --lr 2e-3 \
    --batch_size 32 \
    --accumulate_steps 1 \
    --outer_iters 10 \
    --val_iters 0 \
    --inner_iters 0 \
    --use_sam_pe \
    --enable_clicks \
    --prompt_encoder_checkpoint ckpt/sam_vit_b_01ec64.pth \
    --image_feature_cache_dir data/cache \
    --image_encoder sam_vitb \
    --dtype bfloat16 \
    --dataset cocolvis \
    --dl_workers 16 \
    --prefetch_factor 2

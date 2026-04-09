#!/bin/bash
# Train SimpleMaskVqvae on COCONut HF dataset
# COCONut contains more diverse segmentation masks including stuff classes

N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_simple_mask_vqvae_coconut_ep30_12realddp}
export MASTER_PORT=${3:-29500}
export OMP_NUM_THREADS=4

if [ $N_NODE -eq 0 ]; then
    python train_scripts/train_simple_mask_vqvae.py \
    --out_dir $OUTDIR \
    --outer_iters 10 \
    --inner_iters 0 \
    --val_iters 0 \
    --batch_size 16 \
    --learning_rate 2e-4 \
    --accumulate_steps 1 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --dataset coconut_hf \
    --sam_checkpoint_path ckpt/mobile_sam.pt \
    --dtype bfloat16 \
    --no_compile \
    --loss dicebce \
    --freeze_image_encoder
else
    torchrun --nproc_per_node=$N_NODE train_scripts/train_simple_mask_vqvae.py \
     --out_dir $OUTDIR \
    --outer_iters 10 \
    --inner_iters 0 \
    --val_iters 0 \
    --batch_size 16 \
    --learning_rate 2e-4 \
    --accumulate_steps 1 \
    --num_workers 16 \
    --prefetch_factor 2 \
    --dataset coconut_hf \
    --sam_checkpoint_path ckpt/mobile_sam.pt \
    --dtype bfloat16 \
    --no_compile \
    --loss dicebce \
    --freeze_image_encoder \
    --log_interval 256
fi
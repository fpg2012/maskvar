#!/bin/bash
# Train SimpleMaskAR on COCONut HF dataset using a frozen SimpleMaskVqvae tokenizer/decoder.
# Smoketest example:
#   python train_scripts/train_simple_mask_ar.py --debug_smoketest ...

N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_simple_mask_ar_coconut_ep10}
VQVAE_CKPT=${3:-out/ddp_simple_mask_vqvae_coconut_ep5_19bugfix_retrain/checkpoints/latest.pth}
export MASTER_PORT=${4:-29500}
export OMP_NUM_THREADS=4

if [ $N_NODE -eq 0 ]; then
    python train_scripts/train_simple_mask_ar.py \
    --out_dir $OUTDIR \
    --vqvae_checkpoint $VQVAE_CKPT \
    --vqvae_config simple_mask_vqvae_dim384 \
    --vqvae_image_encoder_checkpoint ckpt/dino_v3_vits.safetensors \
    --vqvae_image_encoder_config dino_v3_vits \
    --outer_iters 10 \
    --inner_iters 0 \
    --val_iters 0 \
    --batch_size 16 \
    --learning_rate 2e-4 \
    --accumulate_steps 1 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --dataset coconut_hf \
    --config simple_mask_ar \
    --dtype bfloat16 \
    --train_subset_index data/subset/coconut_hf_train-25_percent.npy \
    --log_interval 128 \
    --no_compile \
    # --disable_find_unused_parameters
    # --resume_from path/to/checkpoint.pth \
else
    torchrun --nproc_per_node=$N_NODE train_scripts/train_simple_mask_ar.py \
    --out_dir $OUTDIR \
    --vqvae_checkpoint $VQVAE_CKPT \
    --vqvae_config simple_mask_vqvae_dim384 \
    --vqvae_image_encoder_checkpoint ckpt/dino_v3_vits.safetensors \
    --vqvae_image_encoder_config dino_v3_vits \
    --outer_iters 10 \
    --inner_iters 0 \
    --val_iters 0 \
    --batch_size 16 \
    --learning_rate 2e-4 \
    --accumulate_steps 1 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --dataset coconut_hf \
    --config simple_mask_ar \
    --dtype bfloat16 \
    --train_subset_index data/subset/coconut_hf_train-25_percent.npy \
    --log_interval 128 \
    --disable_find_unused_parameters \
    --resume_from $OUTDIR/checkpoints/iter_5968.pth \
    # --resume_from path/to/checkpoint.pth \
    # --no_compile \
fi

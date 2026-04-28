#!/bin/bash
# Train SimpleMaskLatentDiT on frozen SimpleMaskVAEV2 latents

N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_simple_mask_latent_dit_coconut_ep10}
export MASTER_PORT=${3:-29500}
VAE_CKPT=${4:-out/ddp_simple_mask_vae_v2_coconut_ep10/checkpoints/latest.pth}
export OMP_NUM_THREADS=4

if [ $N_NODE -eq 0 ]; then
    python train_scripts/train_simple_mask_diffusion.py \
    --out_dir $OUTDIR \
    --outer_iters 10 \
    --inner_iters 0 \
    --val_iters 0 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --accumulate_steps 1 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --dataset coconut_hf \
    --config simple_mask_latent_dit \
    --vae_config simple_mask_vae_v2_dim384 \
    --vae_checkpoint $VAE_CKPT \
    --vae_image_encoder_checkpoint ckpt/dino_v3_vits.safetensors \
    --vae_image_encoder_config dino_v3_vits \
    --dtype bfloat16 \
    --sample_val_batches 2 \
    --sample_steps 50 \
    --no_compile \
    --train_subset_index data/subset/coconut_hf_train-25_percent.npy
else
    torchrun --nproc_per_node=$N_NODE train_scripts/train_simple_mask_diffusion.py \
    --out_dir $OUTDIR \
    --outer_iters 10 \
    --inner_iters 0 \
    --val_iters 0 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --accumulate_steps 1 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --dataset coconut_hf \
    --config simple_mask_latent_dit \
    --vae_config simple_mask_vae_v2_dim384 \
    --vae_checkpoint $VAE_CKPT \
    --vae_image_encoder_checkpoint ckpt/dino_v3_vits.safetensors \
    --vae_image_encoder_config dino_v3_vits \
    --dtype bfloat16 \
    --sample_val_batches 2 \
    --sample_steps 50 \
    --no_compile \
    --log_interval 64 \
    --train_subset_index data/subset/coconut_hf_train-25_percent.npy \
    --disable_find_unused_parameters
fi

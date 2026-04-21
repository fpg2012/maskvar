#!/bin/bash
# Train SimpleMaskVqvae on COCONut HF dataset
# COCONut contains more diverse segmentation masks including stuff classes

N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_simple_mask_vqvae_coconut_ep5_18vq_kmeans_init}
export MASTER_PORT=${3:-29500}
export OMP_NUM_THREADS=4

if [ $N_NODE -eq 0 ]; then
    python train_scripts/train_simple_mask_vqvae.py \
    --out_dir $OUTDIR \
    --outer_iters 5 \
    --inner_iters 0 \
    --val_iters 0 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --accumulate_steps 1 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --dataset coconut_hf \
    --config simple_mask_vqvae_dim384 \
    --image_encoder_checkpoint ckpt/dino_v3_vits.safetensors \
    --image_encoder_config dino_v3_vits \
    --dtype bfloat16 \
    --no_compile \
    --loss dicenfl \
    --freeze_image_encoder \
    --train_subset_index data/subset/coconut_hf_train-25_percent.npy \
    --enable_vq \
    --vq_loss_weight 1.0 \
    --checkpoint out/ddp_simple_mask_vqvae_coconut_ep10_16pixelshuffle_conv/checkpoints/latest.pth \
    --kmeans_centroids notebooks/out/kmeans_init/kmeans_centroids_n4096.pt
else
    torchrun --nproc_per_node=$N_NODE train_scripts/train_simple_mask_vqvae.py \
    --out_dir $OUTDIR \
    --outer_iters 5 \
    --inner_iters 0 \
    --val_iters 0 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --accumulate_steps 1 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --dataset coconut_hf \
    --config simple_mask_vqvae_dim384 \
    --image_encoder_checkpoint ckpt/dino_v3_vits.safetensors \
    --image_encoder_config dino_v3_vits \
    --dtype bfloat16 \
    --loss dicenfl \
    --freeze_image_encoder \
    --no_compile \
    --log_interval 64 \
    --train_subset_index data/subset/coconut_hf_train-25_percent.npy \
    --enable_vq \
    --vq_loss_weight 1.0 \
    --checkpoint out/ddp_simple_mask_vqvae_coconut_ep10_16pixelshuffle_conv/checkpoints/latest.pth \
    --kmeans_centroids notebooks/out/kmeans_init/kmeans_centroids_n4096.pt
fi
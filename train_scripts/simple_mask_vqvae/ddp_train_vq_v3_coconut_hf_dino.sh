#!/bin/bash
# Fine-tune SimpleMaskVqvaeV3 with VQ enabled, initialized from a pretrained
# V3 checkpoint and KMeans centroids extracted from that checkpoint's latent tokens.

N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_simple_mask_vqvae_vq_v3_coconut_ep5}
export MASTER_PORT=${3:-29500}
BASE_CKPT_DIR=${4:-out/ddp_simple_mask_vqvae_v3_coconut_ep10}
KMEANS_CENTROIDS=${5:-out/kmeans_init/ddp_simple_mask_vqvae_v3_coconut_ep10/kmeans_centroids_n4096.pt}
export OMP_NUM_THREADS=4

if [ "$N_NODE" -eq 0 ]; then
    python train_scripts/train_simple_mask_vqvae.py \
    --out_dir "$OUTDIR" \
    --outer_iters 5 \
    --inner_iters 0 \
    --val_iters 0 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --accumulate_steps 1 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --dataset coconut_hf \
    --config simple_mask_vqvae_v3_dim384 \
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
    --checkpoint "$BASE_CKPT_DIR/checkpoints/latest.pth" \
    --kmeans_centroids "$KMEANS_CENTROIDS"
else
    torchrun --nproc_per_node="$N_NODE" --master_port="$MASTER_PORT" train_scripts/train_simple_mask_vqvae.py \
    --out_dir "$OUTDIR" \
    --outer_iters 5 \
    --inner_iters 0 \
    --val_iters 0 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --accumulate_steps 1 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --dataset coconut_hf \
    --config simple_mask_vqvae_v3_dim384 \
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
    --checkpoint "$BASE_CKPT_DIR/checkpoints/latest.pth" \
    --kmeans_centroids "$KMEANS_CENTROIDS" \
    --disable_find_unused_parameters
fi

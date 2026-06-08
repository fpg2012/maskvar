#!/bin/bash
# Fine-tune PointRopeSAM with PointRend-style point sampling and click-neighborhood priors.
#
# Defaults to the 1-epoch uniform-grid PointRopeSAM checkpoint.
#
# Usage:
#   bash train_scripts/rope_sam/ddp_train_point_pointrend_click_prior_coconut_hf_dino_click.sh [N_NODE] [OUTDIR] [MASTER_PORT] [INIT_CKPT]

N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_rope_sam_point_pointrend_click_prior_coconut_hf_dino_click}
export MASTER_PORT=${3:-29500}
INIT_CKPT=${4:-out/ddp_rope_sam_point_head_uniform_coconut_hf_dino_click/checkpoints/latest.pth}
export OMP_NUM_THREADS=4

COMMON_ARGS=(
    --out_dir "$OUTDIR"
    --outer_iters 3
    --inner_iters 0
    --val_iters 0
    --batch_size 4
    --learning_rate 2e-4
    --accumulate_steps 1
    --num_workers 8
    --prefetch_factor 4
    --dataset coconut_hf
    --config rope_sam_point_head_dim384
    --image_encoder_config dino_v3_vits
    --image_encoder_checkpoint ckpt/dino_v3_vits.safetensors
    --point_sampling_strategy pointrend
    --point_sampling_space output
    --num_points 4096
    --point_rend_coarse_size 16
    --point_rend_max_size 256
    --click_point_radius 2.0
    --click_point_grid_size 5
    --freeze_image_encoder
    --max_clicks 10
    --interactive_click_warmup_iters 3000
    --interactive_stop_iou 0.95
    --click_count_bias_power 2.0
    --loss nfl
    --dtype bfloat16
    --image_size_encoder 1024
    --image_size_mask 256
    --train_subset_index data/subset/coconut_hf_train-25_percent.npy
    --log_interval 128
)

if [ -n "$INIT_CKPT" ]; then
    COMMON_ARGS+=(--checkpoint "$INIT_CKPT")
fi

if [ "$N_NODE" -eq 0 ]; then
    python train_scripts/train_rope_sam.py "${COMMON_ARGS[@]}" --no_compile
else
    torchrun --nproc_per_node="$N_NODE" train_scripts/train_rope_sam.py \
        "${COMMON_ARGS[@]}" \
        --no_compile
fi

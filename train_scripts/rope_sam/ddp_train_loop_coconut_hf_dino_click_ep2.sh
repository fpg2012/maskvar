#!/bin/bash
# Train LoopRopeSAM on COCONut HF dataset with DINO v3 image features and click prompts.
# This is a short 2-epoch run for checking looped decoder behavior.
#
# Usage:
#   bash train_scripts/rope_sam/ddp_train_loop_coconut_hf_dino_click_ep2.sh [N_NODE] [OUTDIR] [MASTER_PORT] [INIT_CKPT] [LOOP_ITERS] [CACHE_DIR]

N_NODE=${1:-4}
OUTDIR=${2:-out/ddp_rope_sam_loop_coconut_hf_dino_click_ep2}
export MASTER_PORT=${3:-29500}
INIT_CKPT=${4:-}
LOOP_ITERS=${5:-4}
CACHE_DIR=${6:-data/cache}
export OMP_NUM_THREADS=4
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}

COMMON_ARGS=(
    --out_dir "$OUTDIR"
    --outer_iters 2
    --inner_iters 0
    --val_iters 32
    --batch_size 16
    --learning_rate 2e-4
    --accumulate_steps 1
    --num_workers 8
    --prefetch_factor 4
    --dataset coconut_hf
    --config rope_sam_loop_dim384
    --loop_block_index -1
    --loop_iters "$LOOP_ITERS"
    --image_encoder_config dino_v3_vits
    --image_encoder_checkpoint ckpt/dino_v3_vits.safetensors
    --image_feature_cache_dir "$CACHE_DIR"
    --image_feature_cache_max_shard 2
    --freeze_image_encoder
    --max_clicks 10
    --interactive_click_warmup_iters 10000
    --loss nfl
    --dtype bfloat16
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
        "${COMMON_ARGS[@]}"
fi

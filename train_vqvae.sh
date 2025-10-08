#!/bin/bash


cat << 'EOF'
Example usage: 

torchrun --nnodes=1 --nproc_per_node=4 \
    --master_addr=127.0.0.1 --master_port=11134 \
    train_vqvae_example.py \
    --num_epochs 50 --batch_size 4 \
    --out_dir out_vqvae_5_stages_v1 \
    --division 1
EOF